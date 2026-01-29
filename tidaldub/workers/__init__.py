"""
Workers Module
==============

Worker framework for the dubbing pipeline.
"""

from .base import BaseWorker, WorkerConfig, ProcessingResult, WorkerPool

__all__ = [
    "BaseWorker",
    "WorkerConfig",
    "ProcessingResult",
    "WorkerPool",
]
