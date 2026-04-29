#!/usr/bin/env python3
"""AI-assisted tuner - uses the model itself as an optimizer."""

import json
import time
import requests
from pathlib import Path

from engines.llama_cpp import LlamaCppEngine, find_available_port, get_available_vram_mb
from tuners.base import BaseTuner


class AITuner(BaseTuner):
    """Model-as-optimizer: let the model propose its own optimal flags."""

    def __init__(self, model_path: str, port: int = None, rounds: int = 8, status_callback=None):
        # Use dynamic ports to avoid conflicts
        if port is None:
            port = find_available_port()
        super().__init__(model_path, port)
        self.rounds = rounds
        self.engine = LlamaCppEngine(model_path, port, status_callback=status_callback)
        self.history = []
        self.status_callback = status_callback
        # AI query server gets its own dynamic port
        self.ai_query_port = find_available_port(start=port + 10)

    def _log(self, msg, level="info"):
        print("[AI-TUNE] [{}] {}".format(level.upper(), msg))
        if self.status_callback:
            self.status_callback(msg, level)

    def run(self) -> dict:
        """Run AI-assisted tuning across multiple rounds."""
        self._log("Starting AI-assisted tuning ({} rounds)".format(self.rounds), "info")
        self._log("This benchmark uses the model itself to propose optimal inference flags.", "info")

        hw_profile = self._get_hardware_profile()
        gpu_name = hw_profile.get('gpu', 'Unknown')
        vram_total = hw_profile.get('vram_total', '?')
        self._log("Hardware: {} (VRAM: {})".format(gpu_name, vram_total), "info")

        model_info = self._get_model_metadata()
        if model_info:
            self._log("Model info: params={}, layers={}".format(
                model_info.get('params', '?'), model_info.get('layers', '?')), "info")

        # Check VRAM before starting AI server
        available_vram = get_available_vram_mb()
        self._log("Available VRAM for AI queries: {:.0f} MB".format(available_vram), "info")
        
        if available_vram < 1024:
            self._log("Insufficient VRAM ({:.0f} MB) for AI query server. Using heuristic fallback.".format(
                available_vram), "warning")
            return self._run_heuristic_fallback(hw_profile, model_info)

        # Start persistent llama-server for AI queries on separate port
        self._log("Starting persistent llama-server on port {} for AI queries...".format(self.ai_query_port), "info")
        ai_engine = LlamaCppEngine(
            self.model_path, port=self.ai_query_port, status_callback=self.status_callback)

        server_started = ai_engine.start_server()
        if not server_started:
            self._log("Failed to start persistent server on port {}. Using heuristic fallback.".format(
                self.ai_query_port), "warning")
            return self._run_heuristic_fallback(hw_profile, model_info)

        self._log("Persistent AI server ready", "success")

        try:
            # Run baseline benchmark using main engine (starts/stops its own server per test)
            self._log("Running baseline benchmark (default flags)...", "info")
            baseline_results = self.engine.benchmark(
                context_lengths=[2048], max_tokens=100)

            baseline_tps = list(baseline_results.values())[0].get("decode_tps", 0)
            self._log("Baseline: {:.1f} tok/s (decode)".format(baseline_tps), "success")
            
            # Log full baseline results for debugging
            self._log("Full baseline results: {}".format(json.dumps(baseline_results)), "info")

            best_tps = baseline_tps
            best_config = {}

            for round_num in range(1, self.rounds + 1):
                self._log("--- Round {}/{} ---".format(round_num, self.rounds), "info")

                prompt = self._build_tuning_prompt(
                    hw_profile=hw_profile, model_info=model_info,
                    baseline_tps=baseline_tps, history=self.history)

                # Log the full prompt for debugging
                self._log("Tuning prompt ({} chars):".format(len(prompt)), "info")
                self._log(prompt[:500] + ("..." if len(prompt) > 500 else ""), "debug")

                self._log("Sending tuning prompt to AI model...", "info")
                suggested_flags = self._ask_model(prompt)

                if not suggested_flags:
                    self._log("Model did not return valid flags, skipping round", "warning")
                    continue

                self._log("AI suggested raw response: {}".format(json.dumps(suggested_flags)), "info")

                # Validate and clean up the suggestion
                cleaned = self._clean_suggestion(suggested_flags)
                self._log("Cleaned suggestion: {}".format(json.dumps(cleaned)), "info")

                threads_val = cleaned.get('threads', 'default')
                batch_val = cleaned.get('batch_size', 'default')
                cache_val = cleaned.get('cache_type', 'default')
                flash_val = cleaned.get('flash_attn', 'not set')
                self._log("Testing: threads={}, batch={}, cache={}, flash={}".format(
                    threads_val, batch_val, cache_val, flash_val), "info")

                test_results = self.engine.benchmark(
                    context_lengths=[2048], max_tokens=100, **cleaned)

                # Log full test results for debugging
                self._log("Full test results: {}".format(json.dumps(test_results)), "debug")

                test_tps = list(test_results.values())[0].get("decode_tps", 0)
                
                if test_tps == 0:
                    self._log("WARNING: Got 0 tok/s from benchmark — server may have crashed or failed to start", "error")
                    # Check for errors in results
                    for ctx, result in test_results.items():
                        if isinstance(result, dict) and "error" in result:
                            self._log("Error in context {}: {}".format(ctx, result["error"]), "error")

                if baseline_tps > 0:
                    improvement = (test_tps - baseline_tps) / baseline_tps * 100
                else:
                    improvement = 0
                self._log("Result: {:.1f} tok/s ({:+.1f}% vs baseline)".format(
                    test_tps, improvement), "success")

                self.history.append({
                    "round": round_num, "flags": cleaned,
                    "tps": test_tps, "improvement": improvement})

                if test_tps > best_tps:
                    best_tps = test_tps
                    best_config = cleaned
                    self._log("** New best! {:.1f} tok/s".format(best_tps), "success")

            if baseline_tps > 0:
                overall_improvement = (best_tps - baseline_tps) / baseline_tps * 100
            else:
                overall_improvement = 0
            self._log("AI tuning complete. Best: {:.1f} tok/s ({:+.1f}% vs baseline)".format(
                best_tps, overall_improvement), "success")

            return {
                "baseline_tps": baseline_tps, "best_tps": best_tps,
                "best_config": best_config, "rounds_history": self.history,
                "improvement_pct": overall_improvement}
        finally:
            self._log("Stopping persistent AI server...", "info")
            ai_engine.stop_server()

    def _clean_suggestion(self, raw: dict) -> dict:
        """Clean and validate AI suggestion before passing to engine."""
        cleaned = {}
        
        # threads - must be positive int
        if 'threads' in raw:
            try:
                t = int(raw['threads'])
                if 1 <= t <= 64:
                    cleaned['threads'] = t
                else:
                    self._log("Thread count {} out of range, using default".format(t), "warning")
            except (ValueError, TypeError):
                self._log("Invalid thread value: {}".format(raw['threads']), "warning")

        # batch_size - must be positive int
        if 'batch_size' in raw:
            try:
                b = int(raw['batch_size'])
                if 64 <= b <= 8192:
                    cleaned['batch_size'] = b
                else:
                    self._log("Batch size {} out of range, using default".format(b), "warning")
            except (ValueError, TypeError):
                self._log("Invalid batch value: {}".format(raw['batch_size']), "warning")

        # cache_type - must be valid type string
        if 'cache_type' in raw:
            ct = str(raw['cache_type']).strip().lower()
            if ct in ('q4_0', 'f16', 'q8_0', 'bf16'):
                cleaned['cache_type'] = ct
            else:
                self._log("Invalid cache type '{}', skipping".format(ct), "warning")

        # flash_attn - normalize to on/off string or omit
        if 'flash_attn' in raw:
            fa = str(raw['flash_attn']).strip().lower()
            if fa in ('on', 'true', 'auto'):
                cleaned['flash_attn'] = 'on'
            elif fa in ('off', 'false'):
                # Don't include — engine won't add the flag
                pass
            else:
                self._log("Invalid flash_attn value '{}', skipping".format(fa), "warning")

        return cleaned

    def _run_heuristic_fallback(self, hw_profile, model_info) -> dict:
        """Fallback when AI server can't start - use heuristic configs."""
        self._log("Running heuristic fallback (no AI server available)...", "warning")

        self._log("Running baseline benchmark...", "info")
        baseline_results = self.engine.benchmark(context_lengths=[2048], max_tokens=100)
        baseline_tps = list(baseline_results.values())[0].get("decode_tps", 0)
        self._log("Baseline: {:.1f} tok/s".format(baseline_tps), "success")

        heuristic_configs = [
            {"threads": 8, "batch_size": 2048, "cache_type": "q4_0"},
            {"threads": 16, "batch_size": 4096, "cache_type": "q4_0"},
            {"threads": 8, "batch_size": 4096, "cache_type": "f16"},
        ]

        best_tps = baseline_tps
        best_config = {}

        for i, config in enumerate(heuristic_configs):
            self._log("Testing heuristic config {}/{}: {}".format(
                i + 1, len(heuristic_configs), json.dumps(config)), "info")
            results = self.engine.benchmark(context_lengths=[2048], max_tokens=100, **config)
            tps = list(results.values())[0].get("decode_tps", 0)
            if baseline_tps > 0:
                imp = (tps - baseline_tps) / baseline_tps * 100
            else:
                imp = 0
            self._log("Result: {:.1f} tok/s ({:+.1f}%)".format(tps, imp), "success")

            self.history.append({"round": i + 1, "flags": config, "tps": tps, "improvement": imp})

            if tps > best_tps:
                best_tps = tps
                best_config = config

        if baseline_tps > 0:
            overall_imp = (best_tps - baseline_tps) / baseline_tps * 100
        else:
            overall_imp = 0
        return {
            "baseline_tps": baseline_tps, "best_tps": best_tps,
            "best_config": best_config, "rounds_history": self.history,
            "improvement_pct": overall_imp}

    def _get_hardware_profile(self) -> dict:
        """Collect hardware profile information."""
        import subprocess

        hw = {"gpu": "Unknown", "vram": "Unknown", "cpu_cores": "Unknown"}

        try:
            result = subprocess.run(
                ["rocm-smi", "-a"],
                capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout
                
                # Parse GPU name and VRAM
                import re as _re
                for line in output.split("\n"):
                    if "Device Name" in line and ":" in line:
                        parts = line.split(":", 1)
                        hw["gpu"] = parts[1].strip() if len(parts) > 1 else "Unknown"
                    
                    # ROCm 4.x+ format: "GPU Memory Allocated (VRAM%): 95"
                    m = _re.search(r'GPU Memory Allocated.*?:\s*(\d+)', line)
                    if m:
                        used_pct = float(m.group(1))
                        total_mb = 24576.0  # RX 7900 XTX default
                        hw["vram_total"] = "{} MB".format(int(total_mb))
                        continue
                    
                    # Older format with slash
                    if ("vram usage" in line.lower()) and "/" in line:
                        try:
                            nums = []
                            for word in line.replace(",", "").split():
                                try:
                                    val = float(word)
                                    nums.append(val)
                                except ValueError:
                                    pass
                            if len(nums) >= 2:
                                hw["vram_total"] = "{} MB".format(int(nums[1]))
                        except (ValueError, IndexError):
                            pass
            else:
                self._log("rocm-smi returned non-zero exit code: {}".format(result.returncode), "warning")
                if result.stderr:
                    self._log("rocm-smi stderr: {}".format(result.stderr[:200]), "debug")
                    
        except FileNotFoundError:
            self._log("rocm-smi not found in PATH", "warning")
        except Exception as e:
            self._log("Could not get GPU info: {}".format(e), "warning")

        return hw

    def _get_model_metadata(self) -> dict:
        """Extract GGUF metadata."""
        import subprocess

        try:
            result = subprocess.run(
                ["llama-llama", "--model", self.model_path, "--verbose"],
                capture_output=True, text=True, timeout=10)

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

        except FileNotFoundError:
            self._log("llama-llama not found, skipping model metadata", "debug")
        except Exception as e:
            self._log("Could not get model metadata: {}".format(e), "debug")
            return {}

    def _build_tuning_prompt(self, hw_profile, model_info, baseline_tps, history):
        """Build the prompt for the AI tuner."""
        lines = []
        lines.append("You are an expert LLM inference optimization specialist.")
        lines.append("")
        lines.append("HARDWARE PROFILE:")
        lines.append(json.dumps(hw_profile, indent=2))
        lines.append("")
        lines.append("MODEL INFO:")
        lines.append(json.dumps(model_info, indent=2))
        lines.append("")
        lines.append("BASELINE PERFORMANCE: {:.1f} tokens/second (decode)".format(baseline_tps))
        lines.append("")
        lines.append("PREVIOUS TUNING ROUNDS:")

        if history:
            for entry in history[-3:]:
                r_num = entry["round"]
                flags_str = json.dumps(entry["flags"])
                tps_val = "{:.1f}".format(entry['tps'])
                imp_val = "{:+.1f}%".format(entry['improvement'])
                lines.append("Round {}: {} -> {} tok/s ({})".format(
                    r_num, flags_str, tps_val, imp_val))
        else:
            lines.append("No previous rounds.")

        lines.append("")
        lines.append("Please suggest optimal llama.cpp server flags to improve decode throughput.")
        lines.append("")
        lines.append('Return ONLY a valid JSON object. Example:')
        lines.append('{')
        lines.append('  "threads": 16,')
        lines.append('  "batch_size": 4096,')
        lines.append('  "cache_type": "q4_0"')
        lines.append('}')
        lines.append("")
        lines.append("Valid keys: threads (int), batch_size (int), cache_type (q4_0 or f16),")
        lines.append('flash_attn (string: "on", "off", or "auto"). Omit any you do not want to change.')
        lines.append("")
        lines.append("Focus on flags that improve decode speed without exceeding VRAM limits.")

        return "\n".join(lines)

    def _ask_model(self, prompt: str) -> dict:
        """Send prompt to the persistent AI server and parse response."""
        request = {
            "prompt": prompt, "n_predict": 200,
            "temperature": 0.1, "stream": False}

        try:
            url = "http://localhost:{}/completion".format(self.ai_query_port)
            self._log("POST {} (timeout=60s)...".format(url), "info")
            response = requests.post(url, json=request, timeout=60)

            if response.status_code == 200:
                data = response.json()
                text = data.get("content", "")
                self._log("Model responded ({} chars)".format(len(text)), "info")
                
                # Log the raw response for debugging
                self._log("Raw model output: {}".format(text[:300].replace("\n", "\\n")), "debug")

                result = self._extract_json(text)
                if not result:
                    preview = text[:200].replace("\n", "\\n")
                    self._log("No JSON found in response. Preview: {}".format(preview), "warning")
                return result
            else:
                self._log("HTTP {}: {}".format(response.status_code, response.text[:200]), "error")

        except requests.exceptions.ConnectionError as e:
            self._log("Connection refused on port {} - server may have crashed".format(
                self.ai_query_port), "error")
        except Exception as e:
            self._log("Error asking model: {}: {}".format(type(e).__name__, e), "error")

        return {}

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from model response."""
        start = text.find('{')
        end = text.rfind('}') + 1

        if start == -1 or end == 0:
            self._log("No curly braces found in response", "warning")
            return {}

        json_str = text[start:end]
        self._log("Extracted JSON candidate: {}".format(json_str[:200]), "debug")

        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
            else:
                self._log("JSON parsed but not a dict: {}".format(type(result).__name__), "warning")
                return {}
        except json.JSONDecodeError as e:
            self._log("JSON parse error: {}".format(e), "warning")
            return {}

    def get_recommendations(self) -> list:
        """Get optimization recommendations."""
        if not self.history:
            return []

        best = max(self.history, key=lambda x: x["tps"])

        return [
            "Best configuration found in round {} with {:.1f} tok/s".format(
                best['round'], best['tps']),
            "Key flags: {}".format(json.dumps(best['flags'])),
            "Improvement over baseline: {:+.1f}%".format(best['improvement'])]
