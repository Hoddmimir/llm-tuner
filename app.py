#!/usr/bin/env python3
"""LLM Tuner - Web Application Backend."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hardware.detector import HardwareDetector
from database import DatabaseManager
from engines.llama_cpp import LlamaCppEngine
from benchlog.benchmark_logger import logger as file_logger


# --- Models ---

class BenchmarkRequest(BaseModel):
    model_path: str
    engine: str = "llama_cpp"
    benchmark_type: str = "quick"  # quick, full, ai_tune, grid_search
    context_lengths: list[int] | None = None
    max_tokens: int = 100
    temperature: float = 0.7
    rounds: int = 8  # for AI tune
    custom_flags: str = ""  # extra engine flags


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
settings_path: Path | None = None

# Real-time status tracking
benchmark_statuses: dict[str, dict] = {}  # id -> {status, messages, progress}
statuses_lock = Lock()

# Cancellation tracking
cancel_requests: dict[str, bool] = {}  # id -> cancelled flag

# Global GPU monitor state
_gpu_data: dict = {"gpu_memory_used_mb": 0, "gpu_memory_total_mb": 24576, "gpu_utilization_pct": 0}
_gpu_lock = Lock()
_gpu_poll_thread: Thread | None = None
_gpu_running = False


def _get_settings() -> dict:
    """Load settings from JSON file."""
    if not settings_path or not settings_path.exists():
        return {"models_directory": "", "theme": "light"}
    try:
        return json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"models_directory": "", "theme": "light"}


def _save_settings(settings: dict):
    """Save settings to JSON file."""
    if settings_path:
        settings_path.write_text(json.dumps(settings, indent=2))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global db, hardware_cache, settings_path, _gpu_running, _gpu_poll_thread
    
    # Initialize database
    db = DatabaseManager()
    db.init()
    
    # Settings file next to the app
    settings_path = Path(__file__).parent / "settings.json"
    
    # Detect hardware on startup
    detector = HardwareDetector()
    hardware_cache = detector.detect()
    
    print(f"LLM Tuner started. Hardware: {hardware_cache.get('vendor', 'unknown')}")
    
    # Start global GPU monitor thread
    _gpu_running = True
    _gpu_poll_thread = Thread(target=_global_gpu_poll, daemon=True)
    _gpu_poll_thread.start()
    print("Global GPU monitor started (1s poll interval)")
    
    yield
    
    # Shutdown
    _gpu_running = False
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


def _emit_status(benchmark_id: str, message: str, level: str = "info", source: str = "app"):
    """Emit a status update for a benchmark — writes to both memory and file."""
    with statuses_lock:
        if benchmark_id not in benchmark_statuses:
            benchmark_statuses[benchmark_id] = {
                "messages": [],
                "progress": 0,
                "status": "pending"
            }
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "level": level
        }
        benchmark_statuses[benchmark_id]["messages"].append(entry)

    # Also write to file-based log
    file_logger.write(benchmark_id, message, level=level, source=source)


def _set_progress(benchmark_id: str, current: int, total: int):
    """Update progress for a benchmark."""
    with statuses_lock:
        if benchmark_id in benchmark_statuses:
            if total > 0:
                benchmark_statuses[benchmark_id]["progress"] = round((current / total) * 100, 1)
            benchmark_statuses[benchmark_id]["current_step"] = current
            benchmark_statuses[benchmark_id]["total_steps"] = total


def _set_status(benchmark_id: str, status: str):
    """Set overall status for a benchmark."""
    with statuses_lock:
        if benchmark_id in benchmark_statuses:
            benchmark_statuses[benchmark_id]["status"] = status


# --- Global GPU Monitor (always-on) ---

def _get_gpu_memory() -> dict:
    """Query GPU memory usage via rocm-smi text output.
    
    ROCm on consumer GPUs doesn't support --json flag reliably, so we parse text output.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "-a"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return _parse_rocm_smi_text(result.stdout)
    except Exception as e:
        print(f"[GPU MONITOR] Error running rocm-smi: {e}")
    
    return {"gpu_memory_used_mb": 0, "gpu_memory_total_mb": 24576, "gpu_utilization_pct": 0}


def _parse_rocm_smi_text(output: str) -> dict:
    """Parse rocm-smi text output for GPU stats.
    
    Handles various ROCm versions and formats.
    """
    used_mb = 0.0
    total_mb = 24576.0  # Default for RX 7900 XTX
    gpu_util = 0.0
    gpu_name = "Unknown"

    lines = output.split("\n")
    
    for line in lines:
        stripped = line.strip()
        
        # GPU name - look for "Device Name:" or similar
        if "Device Name" in stripped and ":" in stripped:
            parts = stripped.split(":", 1)
            gpu_name = parts[1].strip() if len(parts) > 1 else "Unknown"
        
        # VRAM usage patterns:
        # "Vram Usage              : card-0    20534 / 24576 MB (83%)"
        # "VRAM Usage              : 20534 / 24576 MB"
        if ("vram usage" in stripped.lower() or "vram" in stripped.lower()) and "/" in stripped:
            try:
                # Extract numbers around the slash
                parts = stripped.split("/")
                if len(parts) >= 2:
                    # Find all numbers in the line
                    nums = []
                    for word in stripped.replace(",", "").split():
                        try:
                            val = float(word)
                            nums.append(val)
                        except ValueError:
                            pass
                    
                    if len(nums) >= 2:
                        used_mb = nums[0]
                        total_mb = nums[1]
            except (ValueError, IndexError):
                pass
        
        # GPU utilization patterns:
        # "Gpu use%                : card-0    45.3%"
        # "GPU Core Clk Percent" or similar
        if ("gpu use" in stripped.lower() or "gpu usage" in stripped.lower() or 
            "core clk percent" in stripped.lower()) and "%" in stripped:
            try:
                nums = []
                for word in stripped.replace("%", "").replace(",", "").split():
                    try:
                        val = float(word)
                        if 0 <= val <= 100:
                            nums.append(val)
                    except ValueError:
                        pass
                if nums:
                    gpu_util = nums[0]
            except (ValueError, IndexError):
                pass

    return {
        "gpu_memory_used_mb": round(used_mb, 1),
        "gpu_memory_total_mb": round(total_mb, 1),
        "gpu_utilization_pct": round(gpu_util, 1),
        "gpu_name": gpu_name
    }


def _global_gpu_poll():
    """Background thread that polls GPU stats every second — always running."""
    global _gpu_data
    while _gpu_running:
        try:
            data = _get_gpu_memory()
            with _gpu_lock:
                _gpu_data.update(data)
                _gpu_data["timestamp"] = time.time()
        except Exception as e:
            print(f"[GPU MONITOR] Poll error: {e}")
        time.sleep(1)


def get_global_gpu_data() -> dict:
    """Get the latest GPU data from the global monitor."""
    with _gpu_lock:
        return dict(_gpu_data)


# --- SSE Endpoint ---

@app.get("/api/benchmarks/{benchmark_id}/stream")
async def stream_status(benchmark_id: str):
    """Server-Sent Events for real-time benchmark progress + logs."""
    
    async def event_generator():
        last_count = 0
        
        while True:
            with statuses_lock:
                status_data = benchmark_statuses.get(benchmark_id, {})
                current_count = len(status_data.get("messages", []))
            
            # Send new messages since last check
            if current_count > last_count:
                msg_list = status_data.get("messages", [])
                for msg in msg_list[last_count:]:
                    yield f"data: {json.dumps(msg)}\n\n"
                last_count = current_count
            
            # Always include GPU data in progress updates
            progress = status_data.get("progress", 0)
            status = status_data.get("status", "pending")
            
            if progress > 0 or status in ("completed", "failed", "cancelled"):
                gpu_info = get_global_gpu_data()
                progress_msg = {
                    'type': 'progress',
                    'value': progress,
                    'status': status,
                    'gpu_memory_used_mb': gpu_info.get('gpu_memory_used_mb', 0),
                    'gpu_memory_total_mb': gpu_info.get('gpu_memory_total_mb', 24576),
                    'gpu_utilization_pct': gpu_info.get('gpu_utilization_pct', 0),
                }
                step = status_data.get('current_step', 0)
                total = status_data.get('total_steps', '?')
                if isinstance(total, int):
                    progress_msg['current_context'] = f"{step}/{total}"
                yield f"data: {json.dumps(progress_msg)}\n\n"
            
            # Check if benchmark is done
            if status in ("completed", "failed", "cancelled"):
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# --- API Routes ---

@app.get("/api/hardware")
def get_hardware():
    """Get detected hardware profile."""
    return hardware_cache or {}


@app.get("/api/gpu")
def get_gpu_status():
    """Get current GPU memory and utilization stats (always-on monitor)."""
    return get_global_gpu_data()


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
    
    # Initialize status tracking
    with statuses_lock:
        benchmark_statuses[benchmark_id] = {
            "messages": [],
            "progress": 0,
            "status": "pending",
            "current_step": 0,
            "total_steps": 0
        }
    
    # Log to file
    _emit_status(benchmark_id, f"Starting benchmark: {request.benchmark_type}", "info")
    _emit_status(benchmark_id, f"Model: {Path(request.model_path).name}", "info")
    _emit_status(benchmark_id, f"Config: max_tokens={request.max_tokens}, temp={request.temperature}, rounds={request.rounds}", "info")
    if request.custom_flags.strip():
        _emit_status(benchmark_id, f"Custom flags: {request.custom_flags}", "info")
    
    # Log GPU state at start
    gpu_info = get_global_gpu_data()
    _emit_status(benchmark_id, 
                 f"GPU at start: {gpu_info.get('gpu_memory_used_mb', '?')}MB / {gpu_info.get('gpu_memory_total_mb', '?')}MB used",
                 "info")
    
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
    
    # Add real-time status info
    with statuses_lock:
        live_status = benchmark_statuses.get(benchmark_id)
    
    if live_status:
        result["live"] = {
            "messages": live_status.get("messages", []),
            "progress": live_status.get("progress", 0),
            "status": live_status.get("status", "unknown")
        }
    
    return result


@app.get("/api/benchmarks/{benchmark_id}/status")
def get_benchmark_status(benchmark_id: str):
    """Get just the real-time status for a benchmark."""
    with statuses_lock:
        live = benchmark_statuses.get(benchmark_id)
    
    if not live:
        # Check database as fallback
        result = db.get_benchmark(benchmark_id)
        if result:
            return {"status": result.get("status", "unknown"), "messages": []}
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return {
        "status": live.get("status", "unknown"),
        "progress": live.get("progress", 0),
        "current_step": live.get("current_step", 0),
        "total_steps": live.get("total_steps", 0),
        "messages": live.get("messages", [])
    }


@app.get("/api/benchmarks/{benchmark_id}/logs")
def get_benchmark_logs(benchmark_id: str):
    """Get file-based logs for a benchmark run."""
    entries = file_logger.get_entries(benchmark_id)
    if not entries:
        # Try reading from disk as fallback
        content = file_logger.get_file_contents(benchmark_id)
        return {"benchmark_id": benchmark_id, "entries": [], "file_content": content}
    
    return {
        "benchmark_id": benchmark_id,
        "entries": entries,
        "total_entries": len(entries)
    }


@app.get("/api/logs")
def list_logs():
    """List all available benchmark logs."""
    return file_logger.list_logs()


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


@app.post("/api/benchmarks/{benchmark_id}/cancel")
def cancel_benchmark(benchmark_id: str):
    """Cancel a running benchmark."""
    with statuses_lock:
        status_data = benchmark_statuses.get(benchmark_id, {})
    
    if not status_data:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    current_status = status_data.get("status", "unknown")
    if current_status in ("completed", "failed", "cancelled"):
        return {"cancelled": False, "message": f"Benchmark already {current_status}"}
    
    # Set cancellation flag - will be picked up by the running thread
    cancel_requests[benchmark_id] = True
    
    _emit_status(benchmark_id, "Cancellation requested...", "warning")
    _set_status(benchmark_id, "cancelled")
    
    return {"cancelled": True, "message": "Benchmark cancellation in progress"}


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


# --- Settings API ---

class SettingsUpdate(BaseModel):
    models_directory: str = ""
    theme: str = "light"


@app.get("/api/settings")
def get_settings():
    """Get application settings."""
    return _get_settings()


@app.put("/api/settings")
def update_settings(settings: SettingsUpdate):
    """Update application settings."""
    current = _get_settings()
    current["models_directory"] = settings.models_directory
    current["theme"] = settings.theme
    _save_settings(current)
    return current


# --- Model Discovery API ---

@app.get("/api/models")
def list_models():
    """Scan models directory for .gguf files."""
    settings = _get_settings()
    models_dir = settings.get("models_directory", "")
    
    if not models_dir:
        return {"models": [], "directory": ""}
    
    dir_path = Path(models_dir)
    if not dir_path.is_dir():
        return {"models": [], "directory": models_dir, "error": "Directory not found"}
    
    # Recursively find all .gguf files
    models = []
    for gguf_file in dir_path.rglob("*.gguf"):
        stat = gguf_file.stat()
        models.append({
            "path": str(gguf_file),
            "name": gguf_file.name,
            "relative": str(gguf_file.relative_to(dir_path)),
            "size_mb": round(stat.st_size / (1024 * 1024), 1)
        })
    
    # Sort by name
    models.sort(key=lambda m: m["name"].lower())
    
    return {"models": models, "directory": models_dir}


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
        _set_status(benchmark_id, "running")
        
        engine = LlamaCppEngine(
            request.model_path, 
            status_callback=lambda msg, level="info": _emit_status(benchmark_id, msg, level, source="engine")
        )
        
        # Check for cancellation before starting
        if cancel_requests.get(benchmark_id):
            raise Exception("Benchmark cancelled by user")
        
        results = None
        
        if request.benchmark_type == "quick":
            results = run_quick_benchmark(engine, request, benchmark_id)
        elif request.benchmark_type == "full":
            results = run_full_benchmark(engine, request, benchmark_id)
        elif request.benchmark_type == "ai_tune":
            results = run_ai_tune(benchmark_id, request)
        elif request.benchmark_type == "grid_search":
            results = run_grid_search(benchmark_id, request)
        else:
            raise ValueError(f"Unknown benchmark type: {request.benchmark_type}")
        
        # Check for cancellation after running
        if cancel_requests.get(benchmark_id):
            raise Exception("Benchmark cancelled by user")
        
        # Save results to both DB and file log
        db.update_results(benchmark_id, results, "completed")
        file_logger.save_results(benchmark_id, results)
        _set_status(benchmark_id, "completed")
        _emit_status(benchmark_id, "Benchmark completed successfully", "success")
        
    except Exception as e:
        error_msg = f"Benchmark failed: {e}"
        print(error_msg)
        is_cancelled = "cancelled" in str(e).lower()
        db.update_status(benchmark_id, "failed" if not is_cancelled else "cancelled", error=str(e))
        _set_status(benchmark_id, "failed" if not is_cancelled else "cancelled")
        _emit_status(benchmark_id, error_msg, "error" if not is_cancelled else "warning")


def run_quick_benchmark(engine: LlamaCppEngine, request: BenchmarkRequest, 
                         benchmark_id: str) -> dict:
    """Run quick benchmark with single context length."""
    ctx_len = (request.context_lengths or [2048])[0]
    
    _set_progress(benchmark_id, 1, 1)
    
    return engine.benchmark(
        context_lengths=[ctx_len],
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )


def run_full_benchmark(engine: LlamaCppEngine, request: BenchmarkRequest, 
                        benchmark_id: str) -> dict:
    """Run full benchmark across multiple context lengths."""
    ctx_lens = request.context_lengths or [512, 1024, 2048, 4096]
    
    _set_progress(benchmark_id, 0, len(ctx_lens))
    
    return engine.benchmark(
        context_lengths=ctx_lens,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )


def run_ai_tune(benchmark_id: str, request: BenchmarkRequest) -> dict:
    """Run AI-assisted tuning."""
    from tuners.ai_tuner import AITuner
    
    if cancel_requests.get(benchmark_id):
        raise Exception("Benchmark cancelled by user")
    
    tuner = AITuner(
        request.model_path, 
        rounds=request.rounds,
        status_callback=lambda msg, level="info": _emit_status(benchmark_id, msg, level, source="ai_tuner")
    )
    return tuner.run()


def run_grid_search(benchmark_id: str, request: BenchmarkRequest) -> dict:
    """Run grid search optimization."""
    from tuners.grid_search import GridSearchTuner
    
    if cancel_requests.get(benchmark_id):
        raise Exception("Benchmark cancelled by user")
    
    tuner = GridSearchTuner(
        request.model_path,
        status_callback=lambda msg, level="info": _emit_status(benchmark_id, msg, level, source="grid_search")
    )
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
