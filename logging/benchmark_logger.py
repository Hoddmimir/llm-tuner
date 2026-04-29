#!/usr/bin/env python3
"""File-based benchmark logging with per-run log files."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class BenchmarkLogger:
    """Writes verbose logs to disk for each benchmark run."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self._lock = Lock()
        # In-memory cache for fast reads during active runs
        self._active_logs: dict[str, list[dict]] = {}

    def get_log_path(self, benchmark_id: str) -> Path:
        """Get the log file path for a benchmark."""
        return self.logs_dir / f"{benchmark_id}.log"

    def write(self, benchmark_id: str, message: str, level: str = "info", source: str = "app"):
        """Write a log entry to both file and memory cache.
        
        Args:
            benchmark_id: The benchmark run ID
            message: Log message text
            level: info, success, warning, error
            source: app, engine, server, tuner, gpu
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "ts": timestamp,
            "level": level,
            "source": source,
            "msg": message
        }

        line = f"[{timestamp}] [{level.upper():7s}] [{source:8s}] {message}\n"

        with self._lock:
            # Write to file (append)
            log_path = self.get_log_path(benchmark_id)
            try:
                with open(log_path, "a") as f:
                    f.write(line)
            except OSError as e:
                print(f"[LOG ERROR] Failed to write to {log_path}: {e}")

            # Cache in memory for fast SSE reads
            if benchmark_id not in self._active_logs:
                self._active_logs[benchmark_id] = []
            self._active_logs[benchmark_id].append(entry)

    def get_entries(self, benchmark_id: str, limit: int = 500) -> list[dict]:
        """Get recent log entries for a benchmark from memory cache."""
        with self._lock:
            entries = self._active_logs.get(benchmark_id, [])
            return entries[-limit:]

    def get_file_contents(self, benchmark_id: str) -> str:
        """Read the full log file from disk."""
        log_path = self.get_log_path(benchmark_id)
        if not log_path.exists():
            return ""
        try:
            return log_path.read_text()
        except OSError:
            return ""

    def list_logs(self, limit: int = 50) -> list[dict]:
        """List available benchmark logs sorted by modification time."""
        logs = []
        if not self.logs_dir.exists():
            return logs

        for log_file in sorted(self.logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            if len(logs) >= limit:
                break
            stat = log_file.stat()
            bid = log_file.stem
            # Count lines as rough size indicator
            line_count = sum(1 for _ in open(log_file)) if log_file.exists() else 0
            logs.append({
                "id": bid,
                "file": str(log_file),
                "size_bytes": stat.st_size,
                "lines": line_count,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
            })

        return logs

    def save_results(self, benchmark_id: str, results: dict):
        """Save benchmark results as a separate JSON file."""
        results_path = self.logs_dir / f"{benchmark_id}_results.json"
        try:
            with open(results_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "results": results
                }, f, indent=2)
        except OSError as e:
            print(f"[LOG ERROR] Failed to save results for {benchmark_id}: {e}")

    def cleanup(self, benchmark_id: str):
        """Remove in-memory cache for a completed benchmark."""
        with self._lock:
            self._active_logs.pop(benchmark_id, None)


# Global logger instance
logger = BenchmarkLogger()
