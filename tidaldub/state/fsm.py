"""
File-Based Finite State Machine
===============================

The ground truth for all job state. Uses atomic file operations to ensure
state is never corrupted, even during power failures.

Key features:
- Atomic writes (temp file + fsync + rename)
- Cross-platform file locking (Windows + Unix)
- Full state history for debugging
- Human-readable JSON files
"""

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Cross-platform file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    import msvcrt


class Status(str, Enum):
    """Standard status values for FSM states"""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    DEAD_LETTER = "DEAD_LETTER"
    SKIPPED = "SKIPPED"


@dataclass
class StateEntry:
    """A single entry in the state history"""
    status: str
    timestamp: str
    worker_id: Optional[str] = None
    message: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class JobState:
    """Complete state for a dubbing job"""
    job_id: str
    video_path: str
    status: str = Status.PENDING.value
    created_at: str = ""
    updated_at: str = ""
    
    # Configuration
    audio_languages: list = field(default_factory=list)
    subtitle_languages: list = field(default_factory=list)
    quality_preset: str = "balanced"
    
    # Progress tracking
    current_stage: str = ""
    progress_percent: float = 0.0
    
    # Status history
    status_history: list = field(default_factory=list)
    
    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass 
class SegmentState:
    """State for a single segment (chunk) of processing"""
    segment_id: str
    job_id: str
    stage: str  # transcription, translation, tts, etc.
    
    status: str = Status.PENDING.value
    
    # Segment info
    start_time: float = 0.0
    end_time: float = 0.0
    speaker_id: Optional[str] = None
    language: Optional[str] = None  # For per-language stages
    
    # Processing info
    attempts: int = 0
    last_checkpoint: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    output_paths: list = field(default_factory=list)
    
    # Metrics
    duration_seconds: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    
    # History
    status_history: list = field(default_factory=list)
    
    # Error info
    last_error: Optional[str] = None
    error_traceback: Optional[str] = None


class AtomicFileState:
    """
    File-based state management with atomic operations.
    
    This is the GROUND TRUTH for all state. SQLite and Redis are
    caches that can be rebuilt from these files.
    
    Directory structure:
        state/jobs/{job_id}/
            ├── manifest.json       # Immutable job config
            ├── state.json          # Current job state
            ├── stages/
            │   ├── 01_separation/
            │   │   ├── state.json
            │   │   └── tracks/
            │   │       ├── vocals.state
            │   │       └── ...
            │   ├── 02_transcription/
            │   │   └── chunks/
            │   │       ├── chunk_0000.state
            │   │       └── ...
            │   └── ...
            ├── dlq/                # Dead letter items
            └── history/
                └── events.jsonl    # Append-only log
    """
    
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.state_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Core Atomic Operations
    # =========================================================================
    
    def read_json(self, path: Path) -> Optional[dict]:
        """Read a JSON file, returning None if it doesn't exist"""
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Log corruption, return None
            print(f"WARNING: Failed to read {path}: {e}")
            return None
    
    def write_json_atomic(self, path: Path, data: dict) -> None:
        """
        Atomic write: temp file → fsync → rename
        
        This ensures the file is either completely written or not at all.
        Even a power cut mid-write won't corrupt the state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_path = path.with_suffix('.tmp')
        
        # Write to temp file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force to disk
        
        # Atomic rename
        os.replace(temp_path, path)
        
        # Sync parent directory (extra safety on some filesystems)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            os.fsync(dir_fd)
            os.close(dir_fd)
        except (OSError, PermissionError):
            # Windows may not support directory fsync
            pass
    
    def acquire_lock(self, lock_path: Path, blocking: bool = True) -> Any:
        """
        Acquire an exclusive lock on a file.
        Returns a file handle that must be passed to release_lock.
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, 'w')
        
        if HAS_FCNTL:
            # Unix
            flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_file.fileno(), flags)
        else:
            # Windows
            if blocking:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        
        return lock_file
    
    def release_lock(self, lock_file: Any) -> None:
        """Release a lock acquired with acquire_lock"""
        if lock_file:
            if not HAS_FCNTL:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except:
                    pass
            lock_file.close()
    
    # =========================================================================
    # Job State Operations
    # =========================================================================
    
    def get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a job's state files"""
        return self.jobs_dir / job_id
    
    def create_job(self, job_state: JobState) -> None:
        """
        Create a new job with its initial state.
        Also creates the manifest (immutable job config).
        """
        job_dir = self.get_job_dir(job_state.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stages directory structure
        stages = [
            "01_intake",
            "02_separation", 
            "03_transcription",
            "04_diarization",
            "05_translation",
            "06_tts",
            "07_mixing",
            "08_muxing",
        ]
        for stage in stages:
            (job_dir / "stages" / stage).mkdir(parents=True, exist_ok=True)
        
        # Create other directories
        (job_dir / "dlq").mkdir(parents=True, exist_ok=True)
        (job_dir / "history").mkdir(parents=True, exist_ok=True)
        
        # Write manifest (immutable config)
        manifest = {
            "job_id": job_state.job_id,
            "video_path": job_state.video_path,
            "audio_languages": job_state.audio_languages,
            "subtitle_languages": job_state.subtitle_languages,
            "quality_preset": job_state.quality_preset,
            "created_at": job_state.created_at,
        }
        self.write_json_atomic(job_dir / "manifest.json", manifest)
        
        # Write initial state
        job_state.status_history.append({
            "status": Status.PENDING.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Job created"
        })
        self.write_json_atomic(job_dir / "state.json", asdict(job_state))
    
    def get_job_state(self, job_id: str) -> Optional[JobState]:
        """Load a job's current state"""
        job_dir = self.get_job_dir(job_id)
        state_data = self.read_json(job_dir / "state.json")
        if state_data is None:
            return None
        return JobState(**state_data)
    
    def update_job_state(
        self, 
        job_id: str, 
        new_status: Optional[str] = None,
        current_stage: Optional[str] = None,
        progress_percent: Optional[float] = None,
        error: Optional[str] = None,
        worker_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> JobState:
        """
        Thread-safe update of job state with history tracking.
        Uses file locking to prevent concurrent modifications.
        """
        job_dir = self.get_job_dir(job_id)
        lock_path = job_dir / "state.lock"
        
        lock = self.acquire_lock(lock_path)
        try:
            # Load current state
            state = self.get_job_state(job_id)
            if state is None:
                raise ValueError(f"Job {job_id} not found")
            
            # Update fields
            now = datetime.now(timezone.utc).isoformat()
            state.updated_at = now
            
            if new_status:
                state.status = new_status
                state.status_history.append({
                    "status": new_status,
                    "timestamp": now,
                    "worker_id": worker_id,
                    "message": message,
                })
            
            if current_stage:
                state.current_stage = current_stage
            
            if progress_percent is not None:
                state.progress_percent = progress_percent
            
            if error:
                state.last_error = error
                state.retry_count += 1
            
            # Write updated state
            self.write_json_atomic(job_dir / "state.json", asdict(state))
            
            return state
            
        finally:
            self.release_lock(lock)
    
    # =========================================================================
    # Segment State Operations
    # =========================================================================
    
    def get_segment_path(
        self, 
        job_id: str, 
        stage: str, 
        segment_id: str,
        language: Optional[str] = None
    ) -> Path:
        """Get the path for a segment's state file"""
        job_dir = self.get_job_dir(job_id)
        
        if language:
            return job_dir / "stages" / stage / "languages" / language / f"{segment_id}.state"
        else:
            return job_dir / "stages" / stage / "segments" / f"{segment_id}.state"
    
    def create_segment(self, segment: SegmentState) -> None:
        """Create a new segment state"""
        path = self.get_segment_path(
            segment.job_id, 
            segment.stage, 
            segment.segment_id,
            segment.language
        )
        
        segment.status_history.append({
            "status": Status.PENDING.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Segment created"
        })
        
        self.write_json_atomic(path, asdict(segment))
    
    def get_segment_state(
        self,
        job_id: str,
        stage: str,
        segment_id: str,
        language: Optional[str] = None
    ) -> Optional[SegmentState]:
        """Load a segment's current state"""
        path = self.get_segment_path(job_id, stage, segment_id, language)
        data = self.read_json(path)
        if data is None:
            return None
        return SegmentState(**data)
    
    def update_segment_state(
        self,
        job_id: str,
        stage: str,
        segment_id: str,
        language: Optional[str] = None,
        new_status: Optional[str] = None,
        output_paths: Optional[list] = None,
        error: Optional[str] = None,
        error_traceback: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        memory_peak_mb: Optional[float] = None,
        worker_id: Optional[str] = None,
    ) -> SegmentState:
        """Thread-safe update of segment state"""
        path = self.get_segment_path(job_id, stage, segment_id, language)
        lock_path = path.with_suffix('.lock')
        
        lock = self.acquire_lock(lock_path)
        try:
            data = self.read_json(path)
            if data is None:
                raise ValueError(f"Segment {segment_id} not found")
            
            segment = SegmentState(**data)
            now = datetime.now(timezone.utc).isoformat()
            
            if new_status:
                segment.status = new_status
                segment.status_history.append({
                    "status": new_status,
                    "timestamp": now,
                    "worker_id": worker_id,
                })
                
                if new_status == Status.RUNNING.value:
                    segment.attempts += 1
            
            if output_paths:
                segment.output_paths = output_paths
                # Compute output hash
                segment.output_hash = self._compute_file_hash(output_paths[0]) if output_paths else None
            
            if error:
                segment.last_error = error
                segment.error_traceback = error_traceback
            
            if duration_seconds is not None:
                segment.duration_seconds = duration_seconds
            
            if memory_peak_mb is not None:
                segment.memory_peak_mb = memory_peak_mb
            
            segment.last_checkpoint = now
            
            self.write_json_atomic(path, asdict(segment))
            
            return segment
            
        finally:
            self.release_lock(lock)
    
    # =========================================================================
    # Stage State Operations
    # =========================================================================
    
    def get_stage_state(self, job_id: str, stage: str) -> Optional[dict]:
        """Get the overall state for a processing stage"""
        path = self.get_job_dir(job_id) / "stages" / stage / "state.json"
        return self.read_json(path)
    
    def update_stage_state(
        self,
        job_id: str,
        stage: str,
        status: str,
        progress: Optional[dict] = None,
        message: Optional[str] = None,
    ) -> dict:
        """Update the overall state for a processing stage"""
        path = self.get_job_dir(job_id) / "stages" / stage / "state.json"
        lock_path = path.with_suffix('.lock')
        
        lock = self.acquire_lock(lock_path)
        try:
            data = self.read_json(path) or {
                "status": Status.PENDING.value,
                "status_history": [],
            }
            
            now = datetime.now(timezone.utc).isoformat()
            
            data["status"] = status
            data["updated_at"] = now
            data["status_history"].append({
                "status": status,
                "timestamp": now,
                "message": message,
            })
            
            if progress:
                data["progress"] = progress
            
            self.write_json_atomic(path, data)
            
            return data
            
        finally:
            self.release_lock(lock)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> Optional[str]:
        """Compute SHA256 hash of a file"""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    sha256.update(chunk)
            return f"sha256:{sha256.hexdigest()}"
        except (IOError, OSError):
            return None
    
    def list_jobs(self) -> list[str]:
        """List all job IDs"""
        if not self.jobs_dir.exists():
            return []
        return [d.name for d in self.jobs_dir.iterdir() if d.is_dir()]
    
    def list_incomplete_segments(
        self, 
        job_id: str, 
        stage: str
    ) -> list[SegmentState]:
        """List all segments that are not completed for a stage"""
        job_dir = self.get_job_dir(job_id)
        segments_dir = job_dir / "stages" / stage / "segments"
        
        if not segments_dir.exists():
            return []
        
        incomplete = []
        for state_file in segments_dir.glob("*.state"):
            data = self.read_json(state_file)
            if data and data.get("status") != Status.COMPLETED.value:
                incomplete.append(SegmentState(**data))
        
        return incomplete


# Convenience function for creating state manager
def create_state_manager(config: dict) -> AtomicFileState:
    """Create a state manager from config"""
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    return AtomicFileState(state_dir)
