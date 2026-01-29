"""
TidalDub Orchestrator
=====================

Main orchestrator that coordinates the entire dubbing pipeline.

Responsibilities:
- Job lifecycle management
- Pipeline stage coordination
- Worker spawning and monitoring
- Progress tracking
- Error handling and recovery
"""

import os
import sys
import time
import signal
from pathlib import Path
from typing import Optional, Dict, List
import threading
import yaml

from tidaldub.state import AtomicFileState, JobState, Status
from tidaldub.state.database import StateDatabase
from tidaldub.state.events import GlobalEventLog, JobEventLog, EventType
from tidaldub.queues import QueueManager, QueueName, Task
from tidaldub.recovery import RecoveryManager


class PipelineOrchestrator:
    """
    Orchestrates the dubbing pipeline.
    
    Pipeline stages:
    1. Intake - Extract audio from video
    2. Separation - Separate vocals from music/sfx
    3. Transcription - Speech-to-text
    4. Diarization - Identify speakers
    5. Translation - Translate to target languages
    6. TTS - Synthesize dubbed audio
    7. Mixing - Mix dubbed vocals with background
    8. Muxing - Create final video
    """
    
    STAGES = [
        ("intake", QueueName.INTAKE),
        ("separation", QueueName.SEPARATION),
        ("transcription", QueueName.TRANSCRIPTION),
        ("diarization", QueueName.DIARIZATION),
        ("translation", QueueName.TRANSLATION),
        ("tts", QueueName.TTS),
        ("mixing", QueueName.MIXING),
        ("muxing", QueueName.MUXING),
    ]
    
    def __init__(self, config: dict):
        self.config = config
        
        state_dir = config.get("paths", {}).get("state_dir", "./state")
        
        self.fsm = AtomicFileState(state_dir)
        self.database = StateDatabase(Path(state_dir) / "tidaldub.db")
        self.queue_manager = QueueManager(config)
        self.global_log = GlobalEventLog(state_dir)
        
        self.recovery = RecoveryManager(
            self.fsm, self.database, self.global_log, config
        )
        
        self.running = False
        self.state_dir = Path(state_dir)
        self.data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    
    def start(self):
        """Start the orchestrator"""
        print("TidalDub Orchestrator starting...")
        
        # Run recovery on startup
        recovery_result = self.recovery.recover()
        
        if recovery_result.jobs_recovered > 0:
            print(f"Recovered {recovery_result.jobs_recovered} interrupted jobs")
        
        self.global_log.log_startup()
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        print("Orchestrator ready. Waiting for jobs...")
        
        # Main loop - monitor and coordinate
        while self.running:
            try:
                self._process_completed_stages()
                time.sleep(1)
            except Exception as e:
                print(f"Orchestrator error: {e}")
                time.sleep(5)
    
    def _process_completed_stages(self):
        """
        Check for completed stages and queue the next stage.
        
        With pipeline streaming enabled, we start the next stage
        before the current one fully completes (based on threshold).
        """
        pipeline_config = self.config.get("pipeline", {})
        streaming_enabled = pipeline_config.get("streaming", {}).get("enabled", True)
        stream_threshold = pipeline_config.get("streaming", {}).get("stream_threshold_percent", 50)
        
        # Get all incomplete jobs
        incomplete_jobs = self.database.list_incomplete_jobs()
        
        for job in incomplete_jobs:
            job_state = self.fsm.get_job_state(job.id)
            if job_state is None:
                continue
            
            current_stage = job_state.current_stage
            
            if current_stage:
                stage_state = self.fsm.get_stage_state(job.id, current_stage)
                
                if stage_state:
                    status = stage_state.get("status")
                    
                    # Stage fully completed - queue next
                    if status == Status.COMPLETED.value:
                        self._queue_next_stage(job.id, current_stage)
                        continue
                    
                    # Pipeline streaming: check if we can start next stage early
                    if streaming_enabled and status == Status.RUNNING.value:
                        progress = stage_state.get("progress_percent", 0)
                        
                        if progress >= stream_threshold:
                            # Check if next stage already queued
                            next_stage = self._get_next_stage(current_stage)
                            if next_stage and not self._stage_already_queued(job.id, next_stage):
                                print(f"[Streaming] {current_stage} at {progress}%, starting {next_stage}")
                                self._queue_stage(job.id, next_stage)
    
    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """Get the next stage in the pipeline"""
        stage_order = [
            "01_intake",
            "02_separation",
            "03_transcription",
            "04_diarization",
            "05_translation",
            "06_tts",
            "07_mixing",
            "08_muxing",
        ]
        
        try:
            current_idx = stage_order.index(current_stage)
            if current_idx + 1 < len(stage_order):
                return stage_order[current_idx + 1]
        except ValueError:
            pass
        
        return None
    
    def _stage_already_queued(self, job_id: str, stage: str) -> bool:
        """Check if a stage has already been queued"""
        stage_state = self.fsm.get_stage_state(job_id, stage)
        if stage_state:
            status = stage_state.get("status", "")
            return status in [Status.QUEUED.value, Status.RUNNING.value, Status.COMPLETED.value]
        return False
    
    def _queue_next_stage(self, job_id: str, completed_stage: str):
        """Queue the next pipeline stage after current completes"""
        stage_order = [
            "01_intake",
            "02_separation",
            "03_transcription",
            "04_diarization",
            "05_translation",
            "06_tts",
            "07_mixing",
            "08_muxing",
        ]
        
        try:
            current_idx = stage_order.index(completed_stage)
            if current_idx + 1 < len(stage_order):
                next_stage = stage_order[current_idx + 1]
                self._queue_stage(job_id, next_stage)
        except ValueError:
            pass
    
    def _queue_stage(self, job_id: str, stage: str):
        """Queue tasks for a pipeline stage"""
        job_state = self.fsm.get_job_state(job_id)
        if job_state is None:
            return
        
        job_dir = self.data_dir / "temp" / job_id
        
        # Build payload based on stage
        if stage == "01_intake":
            payload = {"video_path": job_state.video_path}
            queue = QueueName.INTAKE
            
        elif stage == "02_separation":
            payload = {"audio_path": str(job_dir / "intake" / "audio.wav")}
            queue = QueueName.SEPARATION
            
        elif stage == "03_transcription":
            payload = {"vocals_path": str(job_dir / "separation" / "vocals.wav")}
            queue = QueueName.TRANSCRIPTION
            
        elif stage == "04_diarization":
            payload = {
                "audio_path": str(job_dir / "intake" / "audio.wav"),
                "transcript_path": str(job_dir / "transcription" / "transcript.json"),
            }
            queue = QueueName.DIARIZATION
            
        elif stage == "05_translation":
            payload = {
                "transcript_path": str(job_dir / "diarization" / "transcript_diarized.json"),
                "target_languages": job_state.audio_languages,
            }
            queue = QueueName.TRANSLATION
            
        elif stage == "06_tts":
            # Queue one task per language
            for lang in job_state.audio_languages:
                self.queue_manager.enqueue(
                    queue=QueueName.TTS,
                    job_id=job_id,
                    task_type="tts",
                    payload={
                        "translated_transcript_path": str(job_dir / "translation" / f"transcript_{lang}.json"),
                        "speaker_map_path": str(job_dir / "diarization" / "speaker_map.json"),
                        "vocals_path": str(job_dir / "separation" / "vocals.wav"),
                        "target_language": lang,
                    }
                )
            return
            
        elif stage == "07_mixing":
            # Queue one task per language
            for lang in job_state.audio_languages:
                self.queue_manager.enqueue(
                    queue=QueueName.MIXING,
                    job_id=job_id,
                    task_type="mixing",
                    payload={
                        "dubbed_vocals_path": str(job_dir / "tts" / lang / f"dubbed_{lang}.wav"),
                        "background_path": str(job_dir / "separation" / "background.wav"),
                        "target_language": lang,
                    }
                )
            return
            
        elif stage == "08_muxing":
            # Collect all audio and subtitle tracks
            audio_tracks = {}
            subtitle_tracks = {}
            
            for lang in job_state.audio_languages:
                audio_tracks[lang] = str(job_dir / "mixed" / f"mixed_{lang}.wav")
            
            for lang in job_state.subtitle_languages:
                subtitle_tracks[lang] = str(job_dir / "subtitles" / f"subtitles_{lang}.srt")
            
            payload = {
                "video_path": job_state.video_path,
                "audio_tracks": audio_tracks,
                "subtitle_tracks": subtitle_tracks,
            }
            queue = QueueName.MUXING
            
        else:
            return
        
        # Update job state
        self.fsm.update_job_state(
            job_id,
            current_stage=stage,
        )
        
        # Queue the task
        self.queue_manager.enqueue(
            queue=queue,
            job_id=job_id,
            task_type=stage,
            payload=payload,
        )
    
    def submit_job(
        self,
        video_path: str,
        audio_languages: List[str],
        subtitle_languages: List[str],
        quality_preset: str = "balanced",
    ) -> str:
        """Submit a new dubbing job"""
        import uuid
        
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Create job state
        job_state = JobState(
            job_id=job_id,
            video_path=video_path,
            audio_languages=audio_languages,
            subtitle_languages=subtitle_languages,
            quality_preset=quality_preset,
        )
        
        # Create in FSM
        self.fsm.create_job(job_state)
        
        # Create in database
        self.database.create_job(
            job_id=job_id,
            video_path=video_path,
            audio_languages=audio_languages,
            subtitle_languages=subtitle_languages,
            quality_preset=quality_preset,
        )
        
        # Log creation
        job_log = JobEventLog(self.state_dir, job_id)
        job_log.log_job_created(video_path, audio_languages, subtitle_languages)
        
        # Queue first stage
        self._queue_stage(job_id, "01_intake")
        
        return job_id
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        print("\nShutting down orchestrator...")
        self.running = False
        self.global_log.log_shutdown()
        self.recovery.remove_pid_file()


def create_orchestrator(config: dict) -> PipelineOrchestrator:
    """Create orchestrator from config"""
    return PipelineOrchestrator(config)


def main():
    """Main entry point"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    orchestrator = create_orchestrator(config)
    orchestrator.start()


if __name__ == "__main__":
    main()
