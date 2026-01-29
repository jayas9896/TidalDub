"""
Transcription Worker
====================

Transcribes audio to text using faster-whisper (CTranslate2 optimized).

Features:
- Auto language detection
- Word-level timestamps
- 50% less VRAM than original Whisper
- Checkpointing for crash recovery

Models (in order of size/quality):
- tiny, base: Fast, lower quality (~1GB VRAM)
- small, medium: Good balance (~2-3GB VRAM)
- large-v3: Best quality (~3GB VRAM with faster-whisper)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tidaldub.workers.base import BaseWorker, WorkerConfig, ProcessingResult
from tidaldub.queues import Task, QueueName

# Lazy imports
WhisperModel = None
torch = None


def lazy_import_faster_whisper():
    """Lazy import faster-whisper (CTranslate2 backend)"""
    global WhisperModel, torch
    if torch is None:
        import torch as _torch
        torch = _torch
    if WhisperModel is None:
        from faster_whisper import WhisperModel as _WhisperModel
        WhisperModel = _WhisperModel


class TranscriptionWorker(BaseWorker):
    """
    Speech-to-text transcription worker using faster-whisper.
    
    Input: Separated vocals audio file
    Output: Timestamped transcript with segments
    
    Uses CTranslate2 backend for ~50% less VRAM than original Whisper.
    RTX 5070 8GB: Can run large-v3 at ~3GB vs 6GB original.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = None
        
        # Get model name from config
        quality_preset = self.global_config.get("quality", {}).get("preset", "balanced")
        presets = self.global_config.get("quality", {}).get("presets", {})
        self.model_name = presets.get(quality_preset, {}).get("whisper_model", "medium")
        self.backend = presets.get(quality_preset, {}).get("whisper_backend", "faster-whisper")
        
        # Chunk duration for long audio
        self.chunk_duration = presets.get(quality_preset, {}).get("chunk_duration_sec", 300)
        
        # Output directory
        self.data_dir = Path(self.global_config.get("paths", {}).get("data_dir", "./data"))
    
    def initialize(self) -> None:
        """Load the Whisper model using faster-whisper backend"""
        lazy_import_faster_whisper()
        
        print(f"[{self.worker_id}] Loading faster-whisper model: {self.model_name}")
        
        # Determine compute type based on GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use int8 quantization for even lower VRAM on 8GB cards
        if device == "cuda":
            # float16 is faster, int8 uses less memory
            compute_type = "float16"  # or "int8_float16" for even lower VRAM
        else:
            compute_type = "int8"
        
        print(f"[{self.worker_id}] Using device: {device}, compute_type: {compute_type}")
        
        # faster-whisper uses CTranslate2 which is much more efficient
        self.model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.global_config.get("paths", {}).get("models_dir", "./models")),
        )
        
        self.device = device
        print(f"[{self.worker_id}] faster-whisper model loaded successfully")
    
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Transcribe vocals audio to text.
        
        Payload:
            vocals_path: Path to separated vocals audio
            job_id: Job ID
        
        Returns:
            ProcessingResult with transcript JSON path
        """
        vocals_path = Path(task.payload.get("vocals_path"))
        job_id = task.job_id
        
        if not vocals_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Vocals file not found: {vocals_path}"
            )
        
        print(f"[{self.worker_id}] Transcribing: {vocals_path}")
        
        try:
            # Check for completed checkpoint
            progress = self.load_progress(task)
            if progress and progress.get("completed"):
                return ProcessingResult(
                    success=True,
                    output_paths=progress.get("output_paths", []),
                    metrics=progress.get("metrics", {}),
                )
            
            start_time = time.time()
            
            # Transcribe with faster-whisper
            print(f"[{self.worker_id}] Running transcription...")
            
            # faster-whisper returns a generator
            segments_gen, info = self.model.transcribe(
                str(vocals_path),
                task="transcribe",
                word_timestamps=True,
                vad_filter=True,  # Voice activity detection for cleaner output
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            # Convert generator to list and process
            transcription_time = time.time() - start_time
            
            # Extract language
            detected_language = info.language
            print(f"[{self.worker_id}] Detected language: {detected_language} (probability: {info.language_probability:.2f})")
            
            # Process segments
            segments = []
            full_text_parts = []
            
            for seg in segments_gen:
                segment_data = {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                
                # Add word-level timestamps if available
                if seg.words:
                    segment_data["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ]
                
                segments.append(segment_data)
                full_text_parts.append(seg.text.strip())
            
            transcription_time = time.time() - start_time
            
            # Calculate audio duration
            audio_duration = info.duration
            
            # Create transcript object
            transcript = {
                "job_id": job_id,
                "source_file": str(vocals_path),
                "language": detected_language,
                "language_probability": info.language_probability,
                "audio_duration_sec": audio_duration,
                "segments": segments,
                "segment_count": len(segments),
                "transcription_time_sec": transcription_time,
                "model": self.model_name,
                "backend": "faster-whisper",
                "full_text": " ".join(full_text_parts),
            }
            
            # Save transcript
            output_dir = self.data_dir / "temp" / job_id / "transcription"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            transcript_path = output_dir / "transcript.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            print(f"[{self.worker_id}] Transcript saved: {transcript_path}")
            print(f"[{self.worker_id}] Segments: {len(segments)}, Duration: {audio_duration:.1f}s")
            
            # Create segment files for downstream processing
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            for seg in segments:
                seg_path = segments_dir / f"segment_{seg['id']:04d}.json"
                with open(seg_path, 'w', encoding='utf-8') as f:
                    json.dump(seg, f, indent=2, ensure_ascii=False)
            
            output_paths = [str(transcript_path)]
            
            metrics = {
                "transcription_time_sec": transcription_time,
                "audio_duration_sec": audio_duration,
                "segment_count": len(segments),
                "detected_language": detected_language,
                "realtime_factor": transcription_time / audio_duration if audio_duration > 0 else 0,
            }
            
            # Save completion checkpoint
            self.save_progress(task, {
                "completed": True,
                "output_paths": output_paths,
                "metrics": metrics,
            })
            
            return ProcessingResult(
                success=True,
                output_paths=output_paths,
                metrics=metrics,
            )
            
        except Exception as e:
            import traceback
            return ProcessingResult(
                success=False,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )
    
    def cleanup(self) -> None:
        """Release GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[{self.worker_id}] Cleanup complete")


def create_transcription_worker(global_config: dict):
    """Factory function to create transcription worker"""
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    from tidaldub.queues import QueueManager
    
    state_dir = global_config.get("paths", {}).get("state_dir", "./state")
    
    config = WorkerConfig(
        worker_type="transcription",
        queue=QueueName.TRANSCRIPTION,
        gpu_device=global_config.get("gpu", {}).get("device", "cuda:0"),
    )
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(global_config)
    
    return TranscriptionWorker(
        config=config,
        fsm=fsm,
        database=database,
        queue_manager=queue_manager,
        global_config=global_config,
    )


if __name__ == "__main__":
    """Run as standalone worker"""
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    worker = create_transcription_worker(config)
    worker.run()
