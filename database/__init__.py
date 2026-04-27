#!/usr/bin/env python3
"""SQLite database manager for benchmark results."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class DatabaseManager:
    """Manage benchmark storage with SQLite."""
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path.home() / "llm_tuner" / "benchmarks.db"
        self.conn = None
    
    def init(self):
        """Initialize database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        
        # Create benchmarks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                engine TEXT NOT NULL,
                benchmark_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                config TEXT,
                results TEXT,
                error TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmarks_status 
            ON benchmarks(status)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmarks_created 
            ON benchmarks(created_at DESC)
        """)
        
        self.conn.commit()
    
    def save_benchmark(self, record: dict):
        """Save a new benchmark record."""
        self.conn.execute("""
            INSERT INTO benchmarks (id, model_path, engine, benchmark_type, status, config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record["id"],
            record["model_path"],
            record["engine"],
            record["benchmark_type"],
            record.get("status", "pending"),
            json.dumps(record.get("config", {})),
            record.get("created_at", datetime.now(timezone.utc).isoformat())
        ))
        
        self.conn.commit()
    
    def get_benchmark(self, benchmark_id: str) -> dict | None:
        """Get a single benchmark by ID."""
        row = self.conn.execute(
            "SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)
        ).fetchone()
        
        if not row:
            return None
        
        result = dict(row)
        
        # Parse JSON fields
        for field in ["config", "results"]:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except Exception:
                    pass
        
        return result
    
    def get_history(self, limit: int = 50) -> list[dict]:
        """Get benchmark history ordered by creation date."""
        rows = self.conn.execute(
            "SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        
        results = []
        for row in rows:
            record = dict(row)
            
            # Parse JSON fields
            for field in ["config", "results"]:
                if record.get(field):
                    try:
                        record[field] = json.loads(record[field])
                    except Exception:
                        pass
            
            results.append(record)
        
        return results
    
    def update_status(self, benchmark_id: str, status: str, error: str = None):
        """Update benchmark status."""
        now = datetime.now(timezone.utc).isoformat()
        
        if status == "running":
            self.conn.execute("""
                UPDATE benchmarks SET status = ?, started_at = ? WHERE id = ?
            """, (status, now, benchmark_id))
        elif status in ("completed", "failed"):
            self.conn.execute("""
                UPDATE benchmarks SET status = ?, completed_at = ?, error = ? WHERE id = ?
            """, (status, now, error or "", benchmark_id))
        
        self.conn.commit()
    
    def update_results(self, benchmark_id: str, results: dict, status: str):
        """Save benchmark results."""
        now = datetime.now(timezone.utc).isoformat()
        
        self.conn.execute("""
            UPDATE benchmarks SET results = ?, status = ?, completed_at = ? WHERE id = ?
        """, (json.dumps(results), status, now, benchmark_id))
        
        self.conn.commit()
    
    def delete_benchmark(self, benchmark_id: str) -> bool:
        """Delete a benchmark record."""
        cursor = self.conn.execute(
            "DELETE FROM benchmarks WHERE id = ?", (benchmark_id,)
        )
        
        deleted = cursor.rowcount > 0
        self.conn.commit()
        
        return deleted
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
