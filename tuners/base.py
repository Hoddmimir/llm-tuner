#!/usr/bin/env python3
"""Base tuner interface."""

from abc import ABC, abstractmethod


class BaseTuner(ABC):
    """Abstract base class for optimization tuners."""
    
    def __init__(self, model_path: str, port: int = 8081):
        self.model_path = model_path
        self.port = port
    
    @abstractmethod
    def run(self) -> dict:
        """Run the tuning process. Returns results with metrics and recommendations."""
        pass
    
    @abstractmethod
    def get_recommendations(self) -> list:
        """Get optimization recommendations based on results."""
        pass
