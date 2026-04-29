#!/usr/bin/env python3
"""Grid search tuner - systematic parameter space exploration."""

import itertools
from engines.llama_cpp import LlamaCppEngine, find_available_port
from tuners.base import BaseTuner


class GridSearchTuner(BaseTuner):
    """Systematic grid search over parameter space."""
    
    def __init__(self, model_path: str, port: int = None, status_callback=None):
        # Use dynamic port to avoid conflicts with other benchmarks
        if port is None:
            port = find_available_port()
        super().__init__(model_path, port)
        self.engine = LlamaCppEngine(model_path, port, status_callback=status_callback)
        self.results = []
        self.status_callback = status_callback
    
    def _log(self, msg, level="info"):
        print(f"[GRID] [{level.upper()}] {msg}")
        if self.status_callback:
            self.status_callback(msg, level)
    
    def run(self) -> dict:
        """Run grid search over parameter combinations."""
        
        # Define parameter space - what we're actually testing
        param_space = {
            "threads": [4, 8, 16],
            "batch_size": [512, 1024, 2048, 4096],
            "cache_type": ["q4_0", "f16"],
        }
        
        # Generate all combinations
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))
        
        self._log(f"Grid search starting: {len(combinations)} configurations to test", "info")
        self._log("Parameters being tested:", "info")
        for key, vals in param_space.items():
            self._log(f"  {key}: {vals}", "info")
        
        best_tps = 0
        best_config = {}
        
        for i, combo in enumerate(combinations, 1):
            config = dict(zip(keys, combo))
            
            self._log(f"[{i}/{len(combinations)}] Testing: threads={config['threads']}, batch={config['batch_size']}, cache={config['cache_type']}", "info")
            
            try:
                results = self.engine.benchmark(
                    context_lengths=[2048],
                    max_tokens=100,
                    **config
                )
                
                tps = list(results.values())[0].get("decode_tps", 0)
                self._log(f"  Result: {tps:.1f} tok/s (decode)", "success")
                
                self.results.append({
                    "config": config,
                    "tps": tps
                })
                
                if tps > best_tps:
                    best_tps = tps
                    best_config = config
                    self._log(f"  ** New best! {tps:.1f} tok/s with {config}", "success")
                    
            except Exception as e:
                self._log(f"  Error testing {config}: {e}", "error")
        
        self._log(f"Grid search complete. Best: {best_tps:.1f} tok/s with {best_config}", "success")
        
        return {
            "total_configs": len(combinations),
            "tested": len(self.results),
            "best_tps": best_tps,
            "best_config": best_config,
            "all_results": self.results
        }
    
    def get_recommendations(self) -> list:
        """Get optimization recommendations."""
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, key=lambda x: x["tps"], reverse=True)
        
        recommendations = [
            f"Best configuration: {sorted_results[0]['config']}",
            f"Performance: {sorted_results[0]['tps']:.1f} tok/s",
            "Top 3 configurations:"
        ]
        
        for i, result in enumerate(sorted_results[:3], 1):
            recommendations.append(f"  {i}. {result['config']} -> {result['tps']:.1f} tok/s")
        
        return recommendations