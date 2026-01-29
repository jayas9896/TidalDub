"""
Append-Only Event Log
=====================

Provides a complete audit trail of all system events.
Uses JSONL format (one JSON object per line) for:
- Easy appending without parsing entire file
- Line-by-line recovery if file is partially corrupted
- Human-readable format for debugging
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(str, Enum):
    """Standard event types"""
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CRASH_DETECTED = "crash_detected"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    
    # Job events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_RESUMED = "job_resumed"
    
    # Stage events
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    
    # Segment events
    SEGMENT_STARTED = "segment_started"
    SEGMENT_COMPLETED = "segment_completed"
    SEGMENT_FAILED = "segment_failed"
    SEGMENT_RETRYING = "segment_retrying"
    
    # Worker events
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    WORKER_HEARTBEAT = "worker_heartbeat"
    
    # DLQ events
    DLQ_ITEM_ADDED = "dlq_item_added"
    DLQ_ITEM_RETRIED = "dlq_item_retried"
    DLQ_ITEM_RESOLVED = "dlq_item_resolved"
    
    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"


@dataclass
class Event:
    """A single event in the log"""
    event_type: str
    timestamp: str
    job_id: Optional[str] = None
    segment_id: Optional[str] = None
    stage: Optional[str] = None
    worker_id: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class EventLog:
    """
    Append-only event log using JSONL format.
    
    This log is NEVER modified after writing, only appended to.
    It provides a complete audit trail that can be used to:
    - Reconstruct system state after crashes
    - Debug issues by replaying events
    - Analyze processing metrics
    
    File structure:
        state/jobs/{job_id}/history/events.jsonl
        
    Each line is a JSON object with timestamp and event data.
    """
    
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(
        self,
        event_type: str | EventType,
        job_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        stage: Optional[str] = None,
        worker_id: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Event:
        """
        Append an event to the log.
        
        Uses atomic append: write + fsync to ensure durability.
        """
        event = Event(
            event_type=str(event_type.value if isinstance(event_type, EventType) else event_type),
            timestamp=datetime.now(timezone.utc).isoformat(),
            job_id=job_id,
            segment_id=segment_id,
            stage=stage,
            worker_id=worker_id,
            message=message,
            data=data,
            duration_ms=duration_ms,
            error=error,
        )
        
        # Convert to JSON, removing None values for compactness
        event_dict = {k: v for k, v in asdict(event).items() if v is not None}
        
        # Atomic append
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_dict, default=str) + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        return event
    
    def read_all(self) -> list[Event]:
        """Read all events from the log"""
        return list(self.iterate())
    
    def iterate(self) -> Iterator[Event]:
        """
        Iterate over all events in the log.
        
        Handles corrupted lines gracefully by skipping them.
        """
        if not self.log_path.exists():
            return
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    yield Event(**data)
                except (json.JSONDecodeError, TypeError) as e:
                    # Log corruption but don't fail
                    print(f"WARNING: Corrupted event at line {line_num}: {e}")
                    continue
    
    def read_last_n(self, n: int) -> list[Event]:
        """Read the last N events (useful for debugging)"""
        # This is O(n) for the whole file, but simple
        # For production, consider using a ring buffer or indexed file
        events = list(self.iterate())
        return events[-n:] if len(events) >= n else events
    
    def find_events(
        self,
        event_type: Optional[str | EventType] = None,
        job_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[Event]:
        """Find events matching criteria"""
        results = []
        
        event_type_str = str(event_type.value if isinstance(event_type, EventType) else event_type) if event_type else None
        
        for event in self.iterate():
            if event_type_str and event.event_type != event_type_str:
                continue
            if job_id and event.job_id != job_id:
                continue
            if segment_id and event.segment_id != segment_id:
                continue
            if since:
                event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                if event_time < since:
                    continue
            
            results.append(event)
        
        return results
    
    def get_job_timeline(self, job_id: str) -> list[Event]:
        """Get all events for a specific job in chronological order"""
        return self.find_events(job_id=job_id)
    
    def get_segment_timeline(self, segment_id: str) -> list[Event]:
        """Get all events for a specific segment"""
        return self.find_events(segment_id=segment_id)


class GlobalEventLog(EventLog):
    """
    System-wide event log for non-job-specific events.
    
    Location: state/system_events.jsonl
    """
    
    def __init__(self, state_dir: str | Path):
        log_path = Path(state_dir) / "system_events.jsonl"
        super().__init__(log_path)
    
    def log_startup(self, worker_count: int = 0, config_hash: Optional[str] = None):
        """Log system startup"""
        self.append(
            EventType.SYSTEM_STARTUP,
            message="TidalDub started",
            data={
                "worker_count": worker_count,
                "config_hash": config_hash,
            }
        )
    
    def log_shutdown(self, reason: str = "normal"):
        """Log system shutdown"""
        self.append(
            EventType.SYSTEM_SHUTDOWN,
            message=f"TidalDub shutdown: {reason}",
        )
    
    def log_crash_detected(self, orphaned_jobs: list[str]):
        """Log that a crash was detected on startup"""
        self.append(
            EventType.CRASH_DETECTED,
            message=f"Detected {len(orphaned_jobs)} jobs in inconsistent state",
            data={"orphaned_jobs": orphaned_jobs},
        )
    
    def log_recovery(self, jobs_recovered: int, segments_requeued: int):
        """Log recovery completion"""
        self.append(
            EventType.RECOVERY_COMPLETED,
            message=f"Recovery complete: {jobs_recovered} jobs, {segments_requeued} segments requeued",
            data={
                "jobs_recovered": jobs_recovered,
                "segments_requeued": segments_requeued,
            }
        )


class JobEventLog(EventLog):
    """
    Per-job event log for detailed job tracking.
    
    Location: state/jobs/{job_id}/history/events.jsonl
    """
    
    def __init__(self, state_dir: str | Path, job_id: str):
        log_path = Path(state_dir) / "jobs" / job_id / "history" / "events.jsonl"
        super().__init__(log_path)
        self.job_id = job_id
    
    def log_job_created(self, video_path: str, audio_langs: list, subtitle_langs: list):
        """Log job creation"""
        self.append(
            EventType.JOB_CREATED,
            job_id=self.job_id,
            message=f"Job created for {video_path}",
            data={
                "video_path": video_path,
                "audio_languages": audio_langs,
                "subtitle_languages": subtitle_langs,
            }
        )
    
    def log_stage_started(self, stage: str, total_segments: int = 0):
        """Log stage start"""
        self.append(
            EventType.STAGE_STARTED,
            job_id=self.job_id,
            stage=stage,
            message=f"Stage {stage} started",
            data={"total_segments": total_segments},
        )
    
    def log_stage_completed(self, stage: str, duration_ms: int):
        """Log stage completion"""
        self.append(
            EventType.STAGE_COMPLETED,
            job_id=self.job_id,
            stage=stage,
            message=f"Stage {stage} completed",
            duration_ms=duration_ms,
        )
    
    def log_segment_started(self, segment_id: str, stage: str, worker_id: str):
        """Log segment processing start"""
        self.append(
            EventType.SEGMENT_STARTED,
            job_id=self.job_id,
            segment_id=segment_id,
            stage=stage,
            worker_id=worker_id,
        )
    
    def log_segment_completed(self, segment_id: str, stage: str, duration_ms: int):
        """Log segment completion"""
        self.append(
            EventType.SEGMENT_COMPLETED,
            job_id=self.job_id,
            segment_id=segment_id,
            stage=stage,
            duration_ms=duration_ms,
        )
    
    def log_segment_failed(self, segment_id: str, stage: str, error: str, retry_count: int):
        """Log segment failure"""
        self.append(
            EventType.SEGMENT_FAILED,
            job_id=self.job_id,
            segment_id=segment_id,
            stage=stage,
            error=error,
            data={"retry_count": retry_count},
        )
    
    def log_dlq_added(self, segment_id: str, task_type: str, error: str):
        """Log item added to dead letter queue"""
        self.append(
            EventType.DLQ_ITEM_ADDED,
            job_id=self.job_id,
            segment_id=segment_id,
            message=f"Added to DLQ: {task_type}",
            error=error,
        )


def create_event_logs(config: dict, job_id: Optional[str] = None):
    """Create event log instances from config"""
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    
    global_log = GlobalEventLog(state_dir)
    
    if job_id:
        job_log = JobEventLog(state_dir, job_id)
        return global_log, job_log
    
    return global_log
