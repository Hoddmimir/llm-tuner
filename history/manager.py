#!/usr/bin/env python3
"""History manager - persistent result storage and comparison."""

import json
from datetime import datetime, timezone
from pathlib import Path


class HistoryManager:
    """Manage benchmark history with JSONL storage."""
    
    def __init__(self, history_dir: str = None):
        self.history_dir = Path(history_dir) if history_dir else Path.home() / "llm_tuner_history"
        self.history_file = self.history_dir / "results.jsonl"
        
        # Create directory if needed
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, result: dict):
        """Save a benchmark result to history."""
        entry = {
            "id": f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            **result
        }
        
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def load_results(self, model_path: str = None) -> list:
        """Load results from history, optionally filtered by model."""
        if not self.history_file.exists():
            return []
        
        results = []
        with open(self.history_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Filter by model if specified
                    if model_path and entry.get("config", {}).get("model") != model_path:
                        continue
                    
                    results.append(entry)
                
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def compare_results(self, results: list = None) -> dict:
        """Compare multiple benchmark results."""
        if not results:
            results = self.load_results()
        
        if not results:
            return {"message": "No results to compare"}
        
        # Group by model and configuration
        comparison = {}
        
        for result in results:
            model = result.get("config", {}).get("model", "Unknown")
            
            if model not in comparison:
                comparison[model] = []
            
            comparison[model].append({
                "timestamp": result.get("timestamp"),
                "tps": result.get("results", {}).get("best_tps", 0),
                "config": result.get("config", {})
            })
        
        return comparison
    
    def get_best_result(self, model_path: str = None) -> dict:
        """Get the best historical result for a model."""
        results = self.load_results(model_path)
        
        if not results:
            return {}
        
        # Find result with highest TPS
        best = max(results, key=lambda x: x.get("results", {}).get("best_tps", 0))
        
        return best
    
    def export_report(self, output_file: str = None) -> str:
        """Export a summary report of all results."""
        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = self.history_dir / f"report_{timestamp}.json"
        
        results = self.load_results()
        
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(results),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(output_file)
    
    def _generate_summary(self, results: list) -> dict:
        """Generate summary statistics."""
        if not results:
            return {}
        
        tps_values = [r.get("results", {}).get("best_tps", 0) for r in results]
        
        return {
            "total_runs": len(results),
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else 0,
            "max_tps": max(tps_values) if tps_values else 0,
            "min_tps": min(tps_values) if tps_values else 0
        }
