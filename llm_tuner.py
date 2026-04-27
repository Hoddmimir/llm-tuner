#!/usr/bin/env python3
"""
LLM Tuner - Inference Engine Optimization Framework

Comprehensive benchmarking and auto-optimization tool for LLM inference engines.
Supports llama.cpp, vLLM, SGLang with AI-assisted tuning and systematic exploration.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from engines.llama_cpp import LlamaCppEngine
from tuners.ai_tuner import AITuner
from tuners.grid_search import GridSearchTuner
from history.manager import HistoryManager


def get_hardware_info():
    """Collect hardware and environment information."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "gpu": None,
        "rocm_version": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # Try to get ROCm info
    try:
        import subprocess
        result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["rocm_version"] = result.stdout.strip()
        
        # Get GPU info
        result = subprocess.run(
            ["rocm-smi", "-a", "--json"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            gpu_data = json.loads(result.stdout)
            if "cards" in gpu_data and len(gpu_data["cards"]) > 0:
                card = gpu_data["cards"][0]
                info["gpu"] = {
                    "name": card.get("card_serial", "Unknown"),
                    "vram_total": card.get("vram_usage", {}).get("total", "Unknown"),
                    "vram_used": card.get("vram_usage", {}).get("used", "Unknown"),
                }
    except Exception as e:
        info["gpu_error"] = str(e)

    return info


def run_benchmark(args):
    """Run benchmark across specified engines."""
    print("=" * 80)
    print("LLM TUNER - Inference Engine Benchmark")
    print("=" * 80)
    
    # Get hardware info
    hw_info = get_hardware_info()
    print(f"\nHardware: {hw_info.get('gpu', {}).get('name', 'Unknown')}")
    print(f"ROCm: {hw_info.get('rocm_version', 'Unknown')}")
    print(f"Model: {args.model}")
    
    # Initialize history manager
    history = HistoryManager()
    
    engines_to_test = []
    if args.engines == "all":
        engines_to_test = ["llama_cpp", "vllm", "sglang"]
    else:
        engines_to_test = [e.strip() for e in args.engines.split(",")]
    
    results = {}
    
    for engine_name in engines_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {engine_name.upper()}")
        print('='*60)
        
        if engine_name == "llama_cpp":
            engine = LlamaCppEngine(args.model, args.port)
            
            # Run baseline benchmark
            if args.context_lengths:
                context_lens = [int(x) for x in args.context_lengths.split(",")]
            else:
                context_lens = [512, 1024, 2048, 4096]
            
            engine_results = engine.benchmark(
                context_lengths=context_lens,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            results[engine_name] = {
                "config": args.__dict__,
                "hardware": hw_info,
                "results": engine_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save to history
            history.save_result(results[engine_name])
            
        else:
            print(f"  Engine {engine_name} not yet implemented")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for engine_name, result in results.items():
        if "results" in result:
            print(f"\n{engine_name.upper()}:")
            for ctx_len, metrics in result["results"].items():
                print(f"  Context {ctx_len}: prefill={metrics.get('prefill_tps', 0):.1f} tok/s, "
                      f"decode={metrics.get('decode_tps', 0):.1f} tok/s")


def run_ai_tune(args):
    """Run AI-assisted tuning."""
    print("=" * 80)
    print("LLM TUNER - AI-Assisted Optimization")
    print("=" * 80)
    
    hw_info = get_hardware_info()
    tuner = AITuner(args.model, args.port, rounds=args.rounds)
    
    results = tuner.run()
    
    # Save to history
    history = HistoryManager()
    history.save_result({
        "tuner": "ai_tune",
        "config": args.__dict__,
        "hardware": hw_info,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Print summary
    print("\n" + "="*80)
    print("AI TUNE SUMMARY")
    print("="*80)
    print(f"Baseline: {results.get('baseline_tps', 0):.1f} tok/s")
    print(f"Tuned:    {results.get('best_tps', 0):.1f} tok/s")
    
    if results.get("baseline_tps", 0) > 0:
        gain = ((results["best_tps"] - results["baseline_tps"]) / results["baseline_tps"]) * 100
        print(f"Gain:     {gain:+.1f}%")


def run_grid_search(args):
    """Run systematic grid search."""
    print("=" * 80)
    print("LLM TUNER - Grid Search Optimization")
    print("=" * 80)
    
    hw_info = get_hardware_info()
    tuner = GridSearchTuner(args.model, args.port)
    
    results = tuner.run()
    
    # Save to history
    history = HistoryManager()
    history.save_result({
        "tuner": "grid_search",
        "config": args.__dict__,
        "hardware": hw_info,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


def main():
    parser = argparse.ArgumentParser(
        description="LLM Tuner - Inference Engine Optimization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with llama.cpp
  python3 llm_tuner.py --model model.gguf --benchmark

  # AI-assisted tuning (8 rounds)
  python3 llm_tuner.py --model model.gguf --ai-tune --rounds 8

  # Grid search over parameters
  python3 llm_tuner.py --model model.gguf --grid-search

  # Compare engines
  python3 llm_tuner.py --model model.gguf --benchmark --engines llama_cpp,vllm
        """
    )
    
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--port", type=int, default=8081, help="Server port (default: 8081)")
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="Run benchmark")
    mode.add_argument("--ai-tune", action="store_true", help="Run AI-assisted tuning")
    mode.add_argument("--grid-search", action="store_true", help="Run grid search")
    
    # Benchmark options
    parser.add_argument("--engines", default="llama_cpp", 
                       help="Engines to test: llama_cpp,vllm,sglang or 'all' (default: llama_cpp)")
    parser.add_argument("--context-lengths", default="512,1024,2048,4096",
                       help="Context lengths to benchmark (comma-separated, default: 512,1024,2048,4096)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Max tokens to generate per test (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    
    # AI tune options
    parser.add_argument("--rounds", type=int, default=8,
                       help="Number of tuning rounds (default: 8)")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args)
    elif args.ai_tune:
        run_ai_tune(args)
    elif args.grid_search:
        run_grid_search(args)


if __name__ == "__main__":
    main()
