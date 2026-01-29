"""
Speaker Diarization Worker
==========================

Identifies who spoke when using pyannote-audio.

Features:
- Speaker segmentation (who spoke when)
- Speaker embedding extraction (voice fingerprints)
- Alignment with transcript segments
- Consistent speaker IDs across the video

Output:
- RTTM file (standard diarization format)
- Speaker embeddings (for TTS voice cloning)
- Annotated transcript with speaker labels
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tidaldub.workers.base import BaseWorker, WorkerConfig, ProcessingResult
from tidaldub.queues import Task, QueueName

# Lazy imports
torch = None
Pipeline = None


def lazy_import():
    """Lazy import heavy dependencies"""
    global torch, Pipeline
    if torch is None:
        import torch as _torch
        torch = _torch
    if Pipeline is None:
        from pyannote.audio import Pipeline as _Pipeline
        Pipeline = _Pipeline


class DiarizationWorker(BaseWorker):
    """
    Speaker diarization worker using pyannote-audio.
    
    Input: Original audio file + transcript
    Output: Speaker-annotated transcript + voice embeddings
    
    This enables:
    - Consistent voice assignment per speaker in TTS
    - Proper attribution in subtitles
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pipeline = None
        self.embedding_model = None
        
        # Output directory
        self.data_dir = Path(self.global_config.get("paths", {}).get("data_dir", "./data"))
    
    def initialize(self) -> None:
        """Load the pyannote diarization pipeline"""
        lazy_import()
        
        print(f"[{self.worker_id}] Loading pyannote diarization pipeline...")
        
        device = self.config.gpu_device if torch.cuda.is_available() else "cpu"
        print(f"[{self.worker_id}] Using device: {device}")
        
        # Load diarization pipeline
        # Note: Requires HuggingFace token for pyannote models
        # User should set HUGGINGFACE_TOKEN environment variable
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        
        if hf_token:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        else:
            print(f"[{self.worker_id}] WARNING: No HuggingFace token found.")
            print(f"[{self.worker_id}] Set HUGGINGFACE_TOKEN env var for pyannote models.")
            # Try to load without token (may fail for gated models)
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                )
            except Exception as e:
                print(f"[{self.worker_id}] Failed to load pipeline: {e}")
                raise
        
        self.pipeline.to(torch.device(device))
        self.device = device
        
        # Load speaker embedding model for voice fingerprints
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            self.embedding_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/spkrec-ecapa-voxceleb",
            )
            print(f"[{self.worker_id}] Speaker embedding model loaded")
        except Exception as e:
            print(f"[{self.worker_id}] Warning: Could not load embedding model: {e}")
            self.embedding_model = None
        
        print(f"[{self.worker_id}] Diarization pipeline loaded successfully")
    
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Perform speaker diarization on audio.
        
        Payload:
            audio_path: Path to original audio
            transcript_path: Path to transcript JSON
            job_id: Job ID
        
        Returns:
            ProcessingResult with speaker-annotated transcript
        """
        audio_path = Path(task.payload.get("audio_path"))
        transcript_path = Path(task.payload.get("transcript_path"))
        job_id = task.job_id
        
        if not audio_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Audio file not found: {audio_path}"
            )
        
        if not transcript_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Transcript file not found: {transcript_path}"
            )
        
        print(f"[{self.worker_id}] Diarizing: {audio_path}")
        
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
            
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # Run diarization
            print(f"[{self.worker_id}] Running speaker diarization...")
            diarization = self.pipeline(str(audio_path))
            
            diarization_time = time.time() - start_time
            
            # Extract speaker segments
            speaker_segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                speaker_segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                })
            
            num_speakers = len(speakers)
            print(f"[{self.worker_id}] Detected {num_speakers} speakers")
            
            # Align with transcript segments
            annotated_segments = self._align_speakers(
                transcript["segments"],
                speaker_segments,
            )
            
            # Extract speaker embeddings (voice fingerprints)
            speaker_embeddings = {}
            if self.embedding_model:
                print(f"[{self.worker_id}] Extracting speaker embeddings...")
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                for speaker in speakers:
                    # Get samples for this speaker
                    speaker_audio = self._extract_speaker_audio(
                        waveform, sample_rate, speaker_segments, speaker
                    )
                    
                    if speaker_audio is not None:
                        # Extract embedding
                        embedding = self.embedding_model.encode_batch(speaker_audio)
                        speaker_embeddings[speaker] = embedding.cpu().numpy().tolist()
            
            # Create output
            output_dir = self.data_dir / "temp" / job_id / "diarization"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save RTTM file (standard diarization format)
            rttm_path = output_dir / "diarization.rttm"
            with open(rttm_path, 'w') as f:
                for seg in speaker_segments:
                    duration = seg["end"] - seg["start"]
                    f.write(f"SPEAKER {job_id} 1 {seg['start']:.3f} {duration:.3f} "
                           f"<NA> <NA> {seg['speaker']} <NA> <NA>\n")
            
            # Save annotated transcript
            annotated_transcript = {
                **transcript,
                "segments": annotated_segments,
                "num_speakers": num_speakers,
                "speakers": list(speakers),
            }
            
            annotated_path = output_dir / "transcript_diarized.json"
            with open(annotated_path, 'w', encoding='utf-8') as f:
                json.dump(annotated_transcript, f, indent=2, ensure_ascii=False)
            
            # Save speaker map (for TTS voice assignment)
            speaker_map = {
                "speakers": list(speakers),
                "embeddings": speaker_embeddings,
                "segment_count_per_speaker": {
                    s: len([seg for seg in annotated_segments if seg.get("speaker") == s])
                    for s in speakers
                },
            }
            
            speaker_map_path = output_dir / "speaker_map.json"
            with open(speaker_map_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_map, f, indent=2)
            
            output_paths = [
                str(annotated_path),
                str(rttm_path),
                str(speaker_map_path),
            ]
            
            metrics = {
                "diarization_time_sec": diarization_time,
                "num_speakers": num_speakers,
                "num_segments": len(annotated_segments),
            }
            
            print(f"[{self.worker_id}] Diarization completed in {diarization_time:.1f}s")
            
            # Save checkpoint
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
    
    def _align_speakers(
        self,
        transcript_segments: List[Dict],
        speaker_segments: List[Dict],
    ) -> List[Dict]:
        """
        Align transcript segments with speaker diarization.
        Uses overlap-based matching.
        """
        annotated = []
        
        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Find overlapping speaker segments
            overlaps = []
            for spk_seg in speaker_segments:
                overlap_start = max(seg_start, spk_seg["start"])
                overlap_end = min(seg_end, spk_seg["end"])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append((spk_seg["speaker"], overlap_duration))
            
            # Assign speaker with most overlap
            if overlaps:
                speaker = max(overlaps, key=lambda x: x[1])[0]
            else:
                speaker = "UNKNOWN"
            
            annotated.append({
                **seg,
                "speaker": speaker,
            })
        
        return annotated
    
    def _extract_speaker_audio(
        self,
        waveform,
        sample_rate: int,
        speaker_segments: List[Dict],
        speaker: str,
        max_duration: float = 30.0,
    ):
        """Extract audio samples for a specific speaker (for embedding)"""
        segments = [s for s in speaker_segments if s["speaker"] == speaker]
        
        if not segments:
            return None
        
        # Collect audio chunks for this speaker
        chunks = []
        total_duration = 0
        
        for seg in segments:
            if total_duration >= max_duration:
                break
            
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            
            chunk = waveform[:, start_sample:end_sample]
            if chunk.shape[-1] > 0:
                chunks.append(chunk)
                total_duration += seg["end"] - seg["start"]
        
        if not chunks:
            return None
        
        # Concatenate chunks
        import torch
        combined = torch.cat(chunks, dim=-1)
        
        # Convert to mono if needed
        if combined.shape[0] > 1:
            combined = combined.mean(dim=0, keepdim=True)
        
        return combined
    
    def cleanup(self) -> None:
        """Release resources"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[{self.worker_id}] Cleanup complete")


def create_diarization_worker(global_config: dict):
    """Factory function to create diarization worker"""
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    from tidaldub.queues import QueueManager
    
    state_dir = global_config.get("paths", {}).get("state_dir", "./state")
    
    config = WorkerConfig(
        worker_type="diarization",
        queue=QueueName.DIARIZATION,
        gpu_device=global_config.get("gpu", {}).get("device", "cuda:0"),
    )
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(global_config)
    
    return DiarizationWorker(
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
    
    worker = create_diarization_worker(config)
    worker.run()
