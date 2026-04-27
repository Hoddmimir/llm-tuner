#!/usr/bin/env python3
"""Base inference engine interface."""

from abc import ABC, abstractmethod


class BaseEngine(ABC):
    """Abstract base class for inference engines."""
    
    def __init__(self, model_path: str, port: int = 8081):
        self.model_path = model_path
        self.port = port
    
    @abstractmethod
    def start_server(self, **kwargs) -> bool:
        """Start the inference server. Returns True if successful."""
        pass
    
    @abstractmethod
    def stop_server(self) -> bool:
        """Stop the inference server. Returns True if successful."""
        pass
    
    @abstractmethod
    def benchmark(self, context_lengths: list = None, **kwargs) -> dict:
        """Run benchmarks across different configurations. Returns metrics dict."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata (parameters, layers, etc.)."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        pass
