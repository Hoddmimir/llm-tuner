#!/usr/bin/env python3
"""Llama.cpp engine backend with HIP/ROCm support."""

import json
import subprocess
import time
import requests
from pathlib import Path

from engines.base import BaseEngine


class LlamaCppEngine(BaseEngine):
    """Backend for llama.cpp inference server with ROCm/HIP support."""
    
    def __init__(self, model_path: str, port: int = 8081, server_binary: str = None):
        super().__init__(model_path, port)
        self.server_binary = server_binary or self._find_server()
        self.process = None
    
    def _find_server(self) -> str:
        """Find llama-server binary."""
        # Check common locations
        candidates = [
            Path("~/projects/llm-server/build/bin/llama-server").expanduser(),
            Path("/usr/local/bin/llama-server"),
            Path("~/bin/llama-server").expanduser(),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # Try to find with which
        try:
            result = subprocess.run(["which", "llama-server"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        raise FileNotFoundError("Could not find llama-server binary")
    
    def start_server(self, **kwargs) -> bool:
        """Start llama.cpp server with specified flags."""
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
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to be ready
            max_wait = 60
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
                time.sleep(1)
            
            print("Warning: Server did not respond to health check within timeout")
            return False
            
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the llama.cpp server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                return True
            except Exception as e:
                print(f"Error stopping server: {e}")
                self.process.kill()
                return False
        return True
    
    def benchmark(self, context_lengths: list = None, max_tokens: int = 100, 
                  temperature: float = 0.7, **kwargs) -> dict:
        """Run benchmarks across different context lengths."""
        if not context_lengths:
            context_lengths = [512, 1024, 2048, 4096]
        
        results = {}
        
        for ctx_len in context_lengths:
            print(f"\nBenchmarking context length: {ctx_len}")
            
            # Start server with this config
            self.start_server(
                context_length=ctx_len,
                **kwargs
            )
            
            time.sleep(2)  # Let server stabilize
            
            try:
                metrics = self._run_single_benchmark(ctx_len, max_tokens, temperature)
                results[ctx_len] = metrics
                
            except Exception as e:
                print(f"Error benchmarking context {ctx_len}: {e}")
                results[ctx_len] = {"error": str(e)}
            
            finally:
                self.stop_server()
        
        return results
    
    def _run_single_benchmark(self, ctx_len: int, max_tokens: int, 
                               temperature: float) -> dict:
        """Run a single benchmark test."""
        # Warmup request
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
            raise Exception(f"Warmup failed: {response.text}")
        
        # Actual benchmark - multiple iterations for averaging
        iterations = 3
        total_prefill_time = 0
        total_decode_time = 0
        
        for i in range(iterations):
            prompt = "Hello, world! " * (ctx_len // 20)  # Approximate token count
            
            benchmark_request = {
                "prompt": prompt[:ctx_len],
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
                
                # Extract timing info from response
                if "timing" in data:
                    total_prefill_time += data["timing"].get("prefill_ms", 0) / 1000
                    total_decode_time += data["timing"].get("predicted_ms", 0) / 1000
        
        # Calculate averages
        avg_prefill = (ctx_len * iterations) / (total_prefill_time or 1)
        avg_decode = (max_tokens * iterations) / (total_decode_time or 1)
        
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
            print(f"Error getting model info: {e}")
        
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
            print(f"Error parsing GGUF: {e}")
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
                if "cards" in data and len(data["cards"]) > 0:
                    card = data["cards"][0]
                    vram = card.get("vram_usage", {})
                    
                    return {
                        "total": vram.get("total", "Unknown"),
                        "used": vram.get("used", "Unknown"),
                        "free": vram.get("available", "Unknown")
                    }
        except Exception as e:
            print(f"Error getting memory usage: {e}")
        
        return {}
