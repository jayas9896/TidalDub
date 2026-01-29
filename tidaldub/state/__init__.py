"""
State Management Module
=======================

Provides three-layer reliability:
1. File-based FSM (ground truth, survives anything)
2. SQLite database (fast queries, rebuildable)
3. Event log (append-only audit trail)
"""

from .fsm import AtomicFileState, JobState, SegmentState
from .database import StateDatabase
from .events import EventLog

__all__ = [
    "AtomicFileState",
    "JobState",
    "SegmentState",
    "StateDatabase",
    "EventLog",
]
