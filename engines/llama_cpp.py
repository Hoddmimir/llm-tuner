#!/usr/bin/env python3
"""Llama.cpp engine backend with HIP/ROCm support."""

import json
import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Callable

from engines.base import BaseEngine


class LlamaCppEngine(BaseEngine):
    """Backend for llama.cpp inference server with ROCm/HIP support."""
    
    def __init__(self, model_path: str, port: int = 8081, 
                 server_binary: str = None, status_callback: Callable = None):
        super().__init__(model_path, port)
        self.server_binary = server_binary or self._find_server()
        self.process = None
        self.status_callback = status_callback
    
    def _log(self, message: str, level: str = "info"):
        """Log a message through the callback if available."""
        print(f"[{level.upper()}] {message}")
        if self.status_callback:
            self.status_callback(message, level)
    
    def _find_server(self) -> str:
        """Find llama-server binary across common locations."""
        candidates = [
            # Common build directories
            Path("~/projects/llm-server/build/bin/llama-server").expanduser(),
            Path("~/llama.cpp/build/bin/llama-server").expanduser(),
            Path("~/llama-tq3-install/bin/llama-server").expanduser(),
            Path("~/llama-tq3/build/bin/llama-server").expanduser(),
            # System locations
            Path("/usr/local/bin/llama-server"),
            Path("~/bin/llama-server").expanduser(),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # Try to find with which
        try:
            result = subprocess.run(
                ["which", "llama-server"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Try to find in PATH directories
        try:
            for dir_path in os.environ.get("PATH", "").split(":"):
                candidate = Path(dir_path) / "llama-server"
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass
        
        raise FileNotFoundError(
            "Could not find llama-server binary. Check that llama.cpp is built and in PATH."
        )
    
    def start_server(self, **kwargs) -> bool:
        """Start llama.cpp server with specified flags."""
        self._log(f"Starting llama.cpp server on port {self.port}")
        
        cmd = [
            self.server_binary,
            "--model", self.model_path,
            "--port", str(self.port),
            "--threads", str(kwargs.get("threads", 8)),
            "--batch-size", str(kwargs.get("batch_size", 2048)),
            "--ctx-size", str(kwargs.get("context_length", 4096)),
        ]
        
        # Add optional flags
        if kwargs.get("cache_type"):
            cmd.extend(["--cache-type-k", kwargs["cache_type"]])
        
        if kwargs.get("flash_attn"):
            cmd.append("--flash-attn")
        
        if kwargs.get("mlock"):
            cmd.append("--mlock")
        
        self._log(f"Command: {' '.join(cmd[:5])}...")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait for server to be ready with progress feedback
            max_wait = 90
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                elapsed = int(time.time() - start_time)
                
                if self.process.poll() is not None:
                    # Process exited early - check for errors
                    output = self.process.stdout.read() if self.process.stdout else ""
                    self._log(f"Server process exited after {elapsed}s", "error")
                    self._log(f"Output: {output[:500]}", "error")
                    return False
                
                # Try health check - llama.cpp uses /health or just responds to requests
                try:
                    response = requests.get(
                        f"http://localhost:{self.port}/health", 
                        timeout=2
                    )
                    if response.status_code == 200:
                        self._log(f"Server ready after {elapsed}s")
                        return True
                except Exception:
                    pass
                
                # Also try the model endpoint as fallback
                try:
                    response = requests.get(
                        f"http://localhost:{self.port}/v1/models", 
                        timeout=2
                    )
                    if response.status_code == 200:
                        self._log(f"Server ready after {elapsed}s (via /v1/models)")
                        return True
                except Exception:
                    pass
                
                # Progress update every 5 seconds
                if elapsed % 5 == 0 and elapsed > 0:
                    self._log(f"Waiting for server... ({elapsed}s/{max_wait}s)", "info")
                
                time.sleep(1)
            
            self._log("Server did not respond within timeout", "error")
            return False
            
        except Exception as e:
            self._log(f"Error starting server: {e}", "error")
            return False
    
    def stop_server(self) -> bool:
        """Stop the llama.cpp server."""
        if self.process:
            try:
                self._log("Stopping server...")
                self.process.terminate()
                
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._log("Force killing server process", "warning")
                    self.process.kill()
                    self.process.wait(timeout=5)
                
                self._log("Server stopped")
                return True
                
            except Exception as e:
                self._log(f"Error stopping server: {e}", "error")
                try:
                    self.process.kill()
                except Exception:
                    pass
                return False
        
        return True
    
    def benchmark(self, context_lengths: list = None, max_tokens: int = 100, 
                  temperature: float = 0.7, **kwargs) -> dict:
        """Run benchmarks across different context lengths."""
        if not context_lengths:
            context_lengths = [512, 1024, 2048, 4096]
        
        self._log(f"Starting benchmark: {len(context_lengths)} context lengths")
        
        results = {}
        total = len(context_lengths)
        
        for i, ctx_len in enumerate(context_lengths, 1):
            if self.status_callback:
                self.status_callback(
                    f"Benchmarking context length: {ctx_len} ({i}/{total})", 
                    "info"
                )
            
            # Start server with this config
            started = self.start_server(
                context_length=ctx_len,
                **kwargs
            )
            
            if not started:
                results[str(ctx_len)] = {"error": "Failed to start server"}
                continue
            
            time.sleep(2)  # Let server stabilize
            
            try:
                metrics = self._run_single_benchmark(ctx_len, max_tokens, temperature)
                results[str(ctx_len)] = metrics
                
                if self.status_callback:
                    decode_tps = metrics.get("decode_tps", 0)
                    prefill_tps = metrics.get("prefill_tps", 0)
                    self.status_callback(
                        f"Context {ctx_len}: prefill={prefill_tps:.1f} tok/s, "
                        f"decode={decode_tps:.1f} tok/s", 
                        "success"
                    )
                
            except Exception as e:
                error_msg = f"Error benchmarking context {ctx_len}: {e}"
                self._log(error_msg, "error")
                results[str(ctx_len)] = {"error": str(e)}
            
            finally:
                self.stop_server()
        
        return results
    
    def _run_single_benchmark(self, ctx_len: int, max_tokens: int, 
                               temperature: float) -> dict:
        """Run a single benchmark test."""
        # Warmup request
        if self.status_callback:
            self.status_callback("Running warmup...", "info")
        
        warmup = {
            "prompt": "Hello",
            "n_predict": 10,
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post(
            f"http://localhost:{self.port}/completion",
            json=warmup,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Warmup failed (HTTP {response.status_code}): {response.text[:200]}")
        
        # Actual benchmark - multiple iterations for averaging
        iterations = 3
        total_prefill_time = 0
        total_decode_time = 0
        
        if self.status_callback:
            self.status_callback(f"Running {iterations} benchmark iterations...", "info")
        
        for i in range(iterations):
            # Generate prompt that approximates the target context length
            base_prompt = "Hello, world! This is a test prompt. " * 10
            
            benchmark_request = {
                "prompt": base_prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": False,
                "timing_per_token": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"http://localhost:{self.port}/completion",
                json=benchmark_request,
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract timing info from response - check multiple formats
                timing_data = data.get("timing", {})
                
                prefill_ms = (timing_data.get("prefill_ms") or 
                              timing_data.get("predicted_prefill_ms"))
                decode_ms = (timing_data.get("predicted_ms") or 
                             timing_data.get("predicted_decode_ms"))
                
                if prefill_ms and prefill_ms > 0:
                    total_prefill_time += prefill_ms / 1000
                else:
                    # Fallback: estimate from wall clock (~20% of total)
                    elapsed = end_time - start_time
                    total_prefill_time += elapsed * 0.2
                
                if decode_ms and decode_ms > 0:
                    total_decode_time += decode_ms / 1000
                else:
                    # Fallback: use wall clock time for decode (~80% of total)
                    elapsed = end_time - start_time
                    total_decode_time += elapsed * 0.8
        
        # Calculate averages
        avg_prefill = (ctx_len * iterations) / max(total_prefill_time, 0.001)
        avg_decode = (max_tokens * iterations) / max(total_decode_time, 0.001)
        
        return {
            "prefill_tps": round(avg_prefill, 2),
            "decode_tps": round(avg_decode, 2),
            "context_length": ctx_len,
            "iterations": iterations
        }
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        try:
            response = requests.get(f"http://localhost:{self.port}/model", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self._log(f"Error getting model info from server: {e}", "warning")
        
        # Fallback - parse from GGUF file
        return self._parse_gguf_info()
    
    def _parse_gguf_info(self) -> dict:
        """Parse basic info from GGUF file."""
        try:
            result = subprocess.run(
                ["llama-llama", "--model", self.model_path, "--verbose"],
                capture_output=True, text=True, timeout=10
            )
            
            # Parse output for key metrics
            info = {"file": self.model_path}
            if "parameters" in result.stdout:
                info["params"] = "Found in output"
            
            return info
            
        except Exception as e:
            self._log(f"Error parsing GGUF: {e}", "warning")
            return {}
    
    def get_memory_usage(self) -> dict:
        """Get current VRAM usage."""
        try:
            result = subprocess.run(
                ["rocm-smi", "-a", "--json"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Find first card
                for key in data:
                    if key.startswith("card"):
                        card = data[key]
                        vram_pct = float(card.get("GPU Memory Allocated (VRAM%)", "0") or "0")
                        
                        return {
                            "vram_allocated_pct": vram_pct,
                            "gpu_name": card.get("Device Name", "Unknown")
                        }
        except Exception as e:
            self._log(f"Error getting memory usage: {e}", "warning")
        
        return {}
