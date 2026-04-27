#!/usr/bin/env python3
"""Grid search tuner - systematic parameter space exploration."""

import itertools
from engines.llama_cpp import LlamaCppEngine
from tuners.base import BaseTuner


class GridSearchTuner(BaseTuner):
    """Systematic grid search over parameter space."""
    
    def __init__(self, model_path: str, port: int = 8081):
        super().__init__(model_path, port)
        self.engine = LlamaCppEngine(model_path, port)
        self.results = []
    
    def run(self) -> dict:
        """Run grid search over parameter combinations."""
        
        # Define parameter space
        param_space = {
            "threads": [4, 8, 16],
            "batch_size": [512, 1024, 2048, 4096],
            "cache_type": ["q4_0", "f16"],
        }
        
        # Generate all combinations
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))
        
        print(f"\nGrid search: {len(combinations)} configurations to test")
        print("="*60)
        
        best_tps = 0
        best_config = {}
        
        for i, combo in enumerate(combinations, 1):
            config = dict(zip(keys, combo))
            
            print(f"\n[{i}/{len(combinations)}] Testing: {config}")
            
            try:
                results = self.engine.benchmark(
                    context_lengths=[2048],
                    max_tokens=100,
                    **config
                )
                
                tps = list(results.values())[0].get("decode_tps", 0)
                print(f"Result: {tps:.1f} tok/s")
                
                self.results.append({
                    "config": config,
                    "tps": tps
                })
                
                if tps > best_tps:
                    best_tps = tps
                    best_config = config
                    
            except Exception as e:
                print(f"Error testing {config}: {e}")
        
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
        
        # Sort by performance
        sorted_results = sorted(self.results, key=lambda x: x["tps"], reverse=True)
        
        recommendations = [
            f"Best configuration: {sorted_results[0]['config']}",
            f"Performance: {sorted_results[0]['tps']:.1f} tok/s",
            "Top 3 configurations:"
        ]
        
        for i, result in enumerate(sorted_results[:3], 1):
            recommendations.append(f"  {i}. {result['config']} -> {result['tps']:.1f} tok/s")
        
        return recommendations
