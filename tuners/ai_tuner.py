#!/usr/bin/env python3
"""AI-assisted tuner - uses the model itself as an optimizer."""

import json
import time
import requests
from pathlib import Path

from engines.llama_cpp import LlamaCppEngine
from tuners.base import BaseTuner


class AITuner(BaseTuner):
    """Model-as-optimizer: let the model propose its own optimal flags."""
    
    def __init__(self, model_path: str, port: int = 8081, rounds: int = 8):
        super().__init__(model_path, port)
        self.rounds = rounds
        self.engine = LlamaCppEngine(model_path, port)
        self.history = []
    
    def run(self) -> dict:
        """Run AI-assisted tuning across multiple rounds."""
        print(f"\nStarting AI-assisted tuning ({self.rounds} rounds)")
        print("="*60)
        
        # Get hardware profile
        hw_profile = self._get_hardware_profile()
        
        # Get model metadata
        model_info = self._get_model_metadata()
        
        # Run baseline benchmark
        print("\nRunning baseline benchmark...")
        baseline_results = self.engine.benchmark(
            context_lengths=[2048],  # Single context for speed
            max_tokens=100
        )
        
        baseline_tps = list(baseline_results.values())[0].get("decode_tps", 0)
        print(f"Baseline: {baseline_tps:.1f} tok/s")
        
        best_tps = baseline_tps
        best_config = {}
        
        # AI tuning rounds
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num}/{self.rounds} ---")
            
            # Build prompt with context
            prompt = self._build_tuning_prompt(
                hw_profile=hw_profile,
                model_info=model_info,
                baseline_tps=baseline_tps,
                history=self.history
            )
            
            # Ask the model for flag suggestions
            suggested_flags = self._ask_model(prompt)
            
            if not suggested_flags:
                print("Model did not return valid flags")
                continue
            
            print(f"Suggested flags: {json.dumps(suggested_flags, indent=2)}")
            
            # Test the suggested configuration
            test_results = self.engine.benchmark(
                context_lengths=[2048],
                max_tokens=100,
                **suggested_flags
            )
            
            test_tps = list(test_results.values())[0].get("decode_tps", 0)
            print(f"Round {round_num} result: {test_tps:.1f} tok/s")
            
            # Track history
            self.history.append({
                "round": round_num,
                "flags": suggested_flags,
                "tps": test_tps,
                "improvement": ((test_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0
            })
            
            # Update best
            if test_tps > best_tps:
                best_tps = test_tps
                best_config = suggested_flags
        
        return {
            "baseline_tps": baseline_tps,
            "best_tps": best_tps,
            "best_config": best_config,
            "rounds_history": self.history,
            "improvement_pct": ((best_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0
        }
    
    def _get_hardware_profile(self) -> dict:
        """Collect hardware profile information."""
        import subprocess
        
        hw = {
            "gpu": "Unknown",
            "vram": "Unknown",
            "cpu_cores": "Unknown"
        }
        
        try:
            # GPU info
            result = subprocess.run(
                ["rocm-smi", "-a", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "cards" in data and len(data["cards"]) > 0:
                    card = data["cards"][0]
                    hw["gpu"] = card.get("card_serial", "Unknown")
                    vram = card.get("vram_usage", {})
                    hw["vram_total"] = vram.get("total", "Unknown")
        
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
        
        return hw
    
    def _get_model_metadata(self) -> dict:
        """Extract GGUF metadata."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["llama-llama", "--model", self.model_path, "--verbose"],
                capture_output=True, text=True, timeout=10
            )
            
            # Parse key info from output
            lines = result.stdout.split('\n')
            metadata = {}
            
            for line in lines:
                if "parameters" in line.lower():
                    metadata["params"] = line.strip()
                elif "layers" in line.lower():
                    metadata["layers"] = line.strip()
                elif "context" in line.lower():
                    metadata["context"] = line.strip()
            
            return metadata
            
        except Exception as e:
            print(f"Warning: Could not get model metadata: {e}")
            return {}
    
    def _build_tuning_prompt(self, hw_profile: dict, model_info: dict, 
                             baseline_tps: float, history: list) -> str:
        """Build the prompt for the AI tuner."""
        
        prompt = f"""You are an expert LLM inference optimization specialist.

HARDWARE PROFILE:
{json.dumps(hw_profile, indent=2)}

MODEL INFO:
{json.dumps(model_info, indent=2)}

BASELINE PERFORMANCE: {baseline_tps:.1f} tokens/second (decode)

PREVIOUS TUNING ROUNDS:
"""
        
        if history:
            for entry in history[-3:]:  # Last 3 rounds for context
                prompt += f"Round {entry['round']}: {json.dumps(entry['flags'])} -> {entry['tps']:.1f} tok/s ({entry['improvement']:+.1f}%)\n"
        else:
            prompt += "No previous rounds.\n"
        
        prompt += """
Please suggest optimal llama.cpp server flags to improve decode throughput.

Return ONLY a JSON object with these possible keys (omit any you don't want to change):
{
  "threads": <int>,
  "batch_size": <int>,
  "cache_type": "<q4_0|f16>",
  "flash_attn": true/false,
  "mlock": true/false
}

Focus on flags that improve decode speed without exceeding VRAM limits."""

        return prompt
    
    def _ask_model(self, prompt: str) -> dict:
        """Send prompt to the model and parse response."""
        
        request = {
            "prompt": prompt,
            "n_predict": 200,
            "temperature": 0.1,  # Low temp for deterministic output
            "stream": False
        }
        
        try:
            response = requests.post(
                f"http://localhost:{self.port}/completion",
                json=request,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("content", "")
                
                # Extract JSON from response
                return self._extract_json(text)
            
        except Exception as e:
            print(f"Error asking model: {e}")
        
        return {}
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from model response."""
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end == 0:
            return {}
        
        json_str = text[start:end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from model response")
            return {}
    
    def get_recommendations(self) -> list:
        """Get optimization recommendations."""
        if not self.history:
            return []
        
        # Find best performing flags
        best = max(self.history, key=lambda x: x["tps"])
        
        recommendations = [
            f"Best configuration found in round {best['round']} with {best['tps']:.1f} tok/s",
            f"Key flags: {json.dumps(best['flags'])}",
            f"Improvement over baseline: {best['improvement']:+.1f}%"
        ]
        
        return recommendations
