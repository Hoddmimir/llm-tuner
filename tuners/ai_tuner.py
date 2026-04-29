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

    def __init__(self, model_path: str, port: int = 8081, rounds: int = 8, status_callback=None):
        super().__init__(model_path, port)
        self.rounds = rounds
        self.engine = LlamaCppEngine(model_path, port, status_callback=status_callback)
        self.history = []
        self.status_callback = status_callback
        self.ai_query_port = 8082

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
        self._log("Hardware: {}".format(gpu_name), "info")

        model_info = self._get_model_metadata()
        if model_info:
            self._log("Model info: params={}, layers={}".format(
                model_info.get('params', '?'), model_info.get('layers', '?')), "info")

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

            best_tps = baseline_tps
            best_config = {}

            for round_num in range(1, self.rounds + 1):
                self._log("--- Round {}/{} ---".format(round_num, self.rounds), "info")

                prompt = self._build_tuning_prompt(
                    hw_profile=hw_profile, model_info=model_info,
                    baseline_tps=baseline_tps, history=self.history)

                self._log("Sending tuning prompt to AI model...", "info")
                suggested_flags = self._ask_model(prompt)

                if not suggested_flags:
                    self._log("Model did not return valid flags, skipping round", "warning")
                    continue

                self._log("AI suggested: {}".format(json.dumps(suggested_flags)), "info")

                threads_val = suggested_flags.get('threads', 'default')
                batch_val = suggested_flags.get('batch_size', 'default')
                cache_val = suggested_flags.get('cache_type', 'default')
                self._log("Testing: threads={}, batch={}, cache={}".format(
                    threads_val, batch_val, cache_val), "info")

                test_results = self.engine.benchmark(
                    context_lengths=[2048], max_tokens=100, **suggested_flags)

                test_tps = list(test_results.values())[0].get("decode_tps", 0)
                if baseline_tps > 0:
                    improvement = (test_tps - baseline_tps) / baseline_tps * 100
                else:
                    improvement = 0
                self._log("Result: {:.1f} tok/s ({:+.1f}% vs baseline)".format(
                    test_tps, improvement), "success")

                self.history.append({
                    "round": round_num, "flags": suggested_flags,
                    "tps": test_tps, "improvement": improvement})

                if test_tps > best_tps:
                    best_tps = test_tps
                    best_config = suggested_flags
                    self._log("** New best! {:.1f} tok/s".format(test_tps), "success")

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
                ["rocm-smi", "-a", "--json"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "cards" in data and len(data["cards"]) > 0:
                    card = data["cards"][0]
                    hw["gpu"] = card.get("card_serial", "Unknown")
                    vram = card.get("vram_usage", {})
                    hw["vram_total"] = vram.get("total", "Unknown")
        except Exception as e:
            print("Warning: Could not get GPU info: {}".format(e))

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

        except Exception as e:
            print("Warning: Could not get model metadata: {}".format(e))
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
            return {}

        json_str = text[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON from model response")
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
