#!/usr/bin/env python3
"""LLM Tuner - Web Application Backend."""

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hardware.detector import HardwareDetector
from database import DatabaseManager
from engines.llama_cpp import LlamaCppEngine


# --- Models ---

class BenchmarkRequest(BaseModel):
    model_path: str
    engine: str = "llama_cpp"
    benchmark_type: str = "quick"  # quick, full, ai_tune, grid_search
    context_lengths: list[int] | None = None
    max_tokens: int = 100
    temperature: float = 0.7
    rounds: int = 8  # for AI tune


class BenchmarkResult(BaseModel):
    id: str
    status: str  # pending, running, completed, failed
    model_path: str
    engine: str
    benchmark_type: str
    results: dict | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


# --- Global State ---

db: DatabaseManager | None = None
hardware_cache: dict | None = None
active_benchmarks: dict[str, dict] = {}  # id -> status info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global db, hardware_cache
    
    # Initialize database
    db = DatabaseManager()
    db.init()
    
    # Detect hardware on startup
    detector = HardwareDetector()
    hardware_cache = detector.detect()
    
    yield
    
    if db:
        db.close()


app = FastAPI(title="LLM Tuner", version="1.0.0", lifespan=lifespan)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- API Routes ---

@app.get("/api/hardware")
def get_hardware():
    """Get detected hardware profile."""
    return hardware_cache or {}


@app.get("/api/engines")
def get_engines():
    """List available engines for this hardware."""
    if not hardware_cache:
        return []
    
    vendor = hardware_cache.get("vendor", "unknown")
    
    # All platforms support llama.cpp
    engines = [{"id": "llama_cpp", "name": "Llama.cpp", "status": "available"}]
    
    # Add platform-specific engines
    if vendor == "rocm":
        engines.append({"id": "vllm_rocm", "name": "vLLM (ROCm)", "status": "experimental"})
        engines.append({"id": "sglang_rocm", "name": "SGLang (ROCm)", "status": "experimental"})
    elif vendor == "cuda":
        engines.append({"id": "vllm_cuda", "name": "vLLM (CUDA)", "status": "available"})
        engines.append({"id": "sglang_cuda", "name": "SGLang (CUDA)", "status": "available"})
    
    return engines


@app.post("/api/benchmarks")
async def start_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a new benchmark run."""
    benchmark_id = str(uuid.uuid4())[:8]
    
    # Validate model path exists
    if not Path(request.model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {request.model_path}")
    
    # Create database record
    record = {
        "id": benchmark_id,
        "model_path": request.model_path,
        "engine": request.engine,
        "benchmark_type": request.benchmark_type,
        "status": "pending",
        "config": request.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    db.save_benchmark(record)
    
    # Start benchmark in background
    background_tasks.add_task(
        run_benchmark, benchmark_id, request
    )
    
    return {"id": benchmark_id, "status": "pending"}


@app.get("/api/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    """Get benchmark status and results."""
    result = db.get_benchmark(benchmark_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return result


@app.get("/api/history")
def get_history(limit: int = 50):
    """Get benchmark history."""
    return db.get_history(limit=limit)


@app.delete("/api/history/{benchmark_id}")
def delete_benchmark(benchmark_id: str):
    """Delete a benchmark record."""
    success = db.delete_benchmark(benchmark_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return {"deleted": True}


@app.get("/api/compare")
def compare_benchmarks(ids: str):
    """Compare multiple benchmark results."""
    benchmark_ids = [x.strip() for x in ids.split(",")]
    benchmarks = []
    
    for bid in benchmark_ids:
        result = db.get_benchmark(bid)
        if result and result.get("results"):
            benchmarks.append(result)
    
    return {
        "benchmarks": benchmarks,
        "comparison": generate_comparison(benchmarks)
    }


def generate_comparison(benchmarks: list[dict]) -> dict:
    """Generate comparison summary across multiple benchmarks."""
    if not benchmarks:
        return {}
    
    # Extract key metrics for comparison
    results = []
    for bm in benchmarks:
        data = bm.get("results", {})
        results.append({
            "id": bm["id"],
            "engine": bm.get("engine"),
            "benchmark_type": bm.get("benchmark_type"),
            "best_tps": data.get("best_tps", 0),
            "baseline_tps": data.get("baseline_tps", 0)
        })
    
    return {"results": results}


def run_benchmark(benchmark_id: str, request: BenchmarkRequest):
    """Run benchmark in background thread."""
    from engines.llama_cpp import LlamaCppEngine
    
    try:
        # Update status to running
        db.update_status(benchmark_id, "running")
        
        engine = LlamaCppEngine(request.model_path)
        
        if request.benchmark_type == "quick":
            results = run_quick_benchmark(engine, request)
        elif request.benchmark_type == "full":
            results = run_full_benchmark(engine, request)
        elif request.benchmark_type == "ai_tune":
            results = run_ai_tune(engine, request)
        elif request.benchmark_type == "grid_search":
            results = run_grid_search(engine, request)
        else:
            raise ValueError(f"Unknown benchmark type: {request.benchmark_type}")
        
        # Save results
        db.update_results(benchmark_id, results, "completed")
        
    except Exception as e:
        db.update_status(benchmark_id, "failed", error=str(e))


def run_quick_benchmark(engine: LlamaCppEngine, request: BenchmarkRequest) -> dict:
    """Run quick benchmark with single context length."""
    ctx_len = (request.context_lengths or [2048])[0]
    
    return engine.benchmark(
        context_lengths=[ctx_len],
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )


def run_full_benchmark(engine: LlamaCppEngine, request: BenchmarkRequest) -> dict:
    """Run full benchmark across multiple context lengths."""
    ctx_lens = request.context_lengths or [512, 1024, 2048, 4096]
    
    return engine.benchmark(
        context_lengths=ctx_lens,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )


def run_ai_tune(engine: LlamaCppEngine, request: BenchmarkRequest) -> dict:
    """Run AI-assisted tuning."""
    from tuners.ai_tuner import AITuner
    
    tuner = AITuner(request.model_path, rounds=request.rounds)
    return tuner.run()


def run_grid_search(engine: LlamaCppEngine, request: BenchmarkRequest) -> dict:
    """Run grid search optimization."""
    from tuners.grid_search import GridSearchTuner
    
    tuner = GridSearchTuner(request.model_path)
    return tuner.run()


# Mount static files for frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_frontend():
    """Serve the web frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>LLM Tuner</h1><p>Frontend not found.</p>")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8090))
    
    print(f"Starting LLM Tuner on port {port}")
    print(f"Hardware: {hardware_cache}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.environ.get("DEV") else False
    )
