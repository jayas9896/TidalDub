"""
Startup Recovery System
=======================

Handles system recovery after crashes, power outages, or unexpected shutdowns.

Recovery process:
1. Scan FSM for all job states
2. Detect jobs that were running when system died (orphaned)
3. Rebuild SQLite from FSM if needed
4. Requeue incomplete segments for processing
5. Resume jobs from last checkpoint
"""

import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from .state import AtomicFileState, StateDatabase, Status
from .state.events import GlobalEventLog, JobEventLog, EventType


@dataclass
class RecoveryResult:
    """Results from recovery process"""
    jobs_found: int = 0
    jobs_recovered: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    segments_requeued: int = 0
    database_rebuilt: bool = False
    errors: list = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class OrphanedJob:
    """A job that was running when system crashed"""
    job_id: str
    status: str
    current_stage: str
    incomplete_segments: list


class RecoveryManager:
    """
    Manages system recovery on startup.
    
    This is called during system initialization to:
    1. Detect if the previous shutdown was unclean
    2. Recover job state from FSM
    3. Rebuild SQLite if corrupted
    4. Prepare incomplete jobs for resumption
    """
    
    def __init__(
        self,
        fsm: AtomicFileState,
        database: StateDatabase,
        global_log: GlobalEventLog,
        config: dict,
    ):
        self.fsm = fsm
        self.database = database
        self.global_log = global_log
        self.config = config
        
        self.state_dir = Path(config.get("paths", {}).get("state_dir", "./state"))
        self.lock_file = self.state_dir / "tidaldub.lock"
        self.pid_file = self.state_dir / "tidaldub.pid"
    
    def recover(self) -> RecoveryResult:
        """
        Main recovery entry point.
        Called on every startup.
        """
        result = RecoveryResult()
        
        # Check for unclean shutdown
        crash_detected = self._detect_crash()
        
        if crash_detected:
            self.global_log.append(
                EventType.CRASH_DETECTED,
                message="Previous shutdown was unclean",
            )
        
        # Scan FSM for all jobs
        job_ids = self.fsm.list_jobs()
        result.jobs_found = len(job_ids)
        
        # Check database integrity
        if self._database_needs_rebuild():
            self.global_log.append(
                EventType.RECOVERY_STARTED,
                message="Rebuilding database from FSM",
            )
            self._rebuild_database()
            result.database_rebuilt = True
        
        # Process each job
        orphaned_jobs = []
        for job_id in job_ids:
            try:
                job_result = self._recover_job(job_id)
                
                if job_result == "recovered":
                    result.jobs_recovered += 1
                    orphaned_jobs.append(job_id)
                elif job_result == "completed":
                    result.jobs_completed += 1
                elif job_result == "failed":
                    result.jobs_failed += 1
                    
            except Exception as e:
                result.errors.append(f"Job {job_id}: {str(e)}")
        
        # Log orphaned jobs if any
        if orphaned_jobs:
            self.global_log.log_crash_detected(orphaned_jobs)
        
        # Requeue incomplete segments
        for job_id in orphaned_jobs:
            segments = self._requeue_incomplete_segments(job_id)
            result.segments_requeued += segments
        
        # Log recovery completion
        self.global_log.log_recovery(
            jobs_recovered=result.jobs_recovered,
            segments_requeued=result.segments_requeued,
        )
        
        # Write PID file for clean shutdown detection
        self._write_pid_file()
        
        return result
    
    def _detect_crash(self) -> bool:
        """
        Detect if the previous shutdown was unclean.
        
        If PID file exists and process is not running, it was a crash.
        """
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                content = f.read().strip()
                parts = content.split(':')
                pid = int(parts[0])
                hostname = parts[1] if len(parts) > 1 else None
            
            # Check if same machine
            if hostname and hostname != socket.gethostname():
                # Different machine, ignore
                return False
            
            # Check if process is still running
            if self._is_process_running(pid):
                # Process is running, might be a race condition
                return False
            
            # Process not running but PID file exists = crash
            return True
            
        except (ValueError, IOError):
            # Corrupted PID file, assume crash
            return True
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running"""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:  # Unix
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False
    
    def _write_pid_file(self) -> None:
        """Write PID file for crash detection"""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, 'w') as f:
            f.write(f"{os.getpid()}:{socket.gethostname()}")
    
    def remove_pid_file(self) -> None:
        """Remove PID file on clean shutdown"""
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass
    
    def _database_needs_rebuild(self) -> bool:
        """Check if database needs to be rebuilt from FSM"""
        try:
            # Try a simple query
            jobs = self.database.list_jobs()
            
            # Compare count with FSM
            fsm_job_count = len(self.fsm.list_jobs())
            db_job_count = len(jobs)
            
            # If counts differ significantly, rebuild
            if abs(fsm_job_count - db_job_count) > 0:
                return True
            
            return False
            
        except Exception:
            # Database error, needs rebuild
            return True
    
    def _rebuild_database(self) -> None:
        """Rebuild SQLite database from FSM"""
        self.database.rebuild_from_fsm(self.fsm)
    
    def _recover_job(self, job_id: str) -> str:
        """
        Recover a single job.
        
        Returns:
            'completed' - Job was already complete
            'failed' - Job was in failed state
            'recovered' - Job was interrupted and needs resumption
        """
        job_state = self.fsm.get_job_state(job_id)
        
        if job_state is None:
            return 'failed'
        
        status = job_state.status
        
        if status == Status.COMPLETED.value:
            return 'completed'
        
        if status == Status.FAILED.value or status == Status.DEAD_LETTER.value:
            return 'failed'
        
        if status in [Status.RUNNING.value, Status.PENDING.value, Status.QUEUED.value]:
            # Job was interrupted
            self._reset_interrupted_segments(job_id)
            
            # Update job status to indicate recovery
            self.fsm.update_job_state(
                job_id,
                message="Recovered after system restart",
            )
            
            # Log to job event log
            job_log = JobEventLog(self.state_dir, job_id)
            job_log.append(
                EventType.JOB_RESUMED,
                job_id=job_id,
                message="Job resumed after system recovery",
            )
            
            return 'recovered'
        
        return 'completed'
    
    def _reset_interrupted_segments(self, job_id: str) -> None:
        """
        Reset segments that were RUNNING when system crashed.
        They need to be reset to PENDING for reprocessing.
        """
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
            incomplete = self.fsm.list_incomplete_segments(job_id, stage)
            
            for segment in incomplete:
                if segment.status == Status.RUNNING.value:
                    # Reset to PENDING
                    self.fsm.update_segment_state(
                        job_id=job_id,
                        stage=stage,
                        segment_id=segment.segment_id,
                        language=segment.language,
                        new_status=Status.PENDING.value,
                    )
                    
                    # Update database too
                    self.database.update_segment_status(
                        segment.segment_id,
                        Status.PENDING.value,
                    )
    
    def _requeue_incomplete_segments(self, job_id: str) -> int:
        """
        Requeue incomplete segments for processing.
        
        Returns the number of segments requeued.
        """
        count = 0
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
            incomplete = self.fsm.list_incomplete_segments(job_id, stage)
            
            for segment in incomplete:
                if segment.status == Status.PENDING.value:
                    # TODO: Add to queue manager when implemented
                    count += 1
        
        return count


def create_recovery_manager(config: dict) -> RecoveryManager:
    """Create recovery manager from config"""
    from .state import AtomicFileState
    from .state.database import StateDatabase
    
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    global_log = GlobalEventLog(state_dir)
    
    return RecoveryManager(fsm, database, global_log, config)
