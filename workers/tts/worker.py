"""
Text-to-Speech Worker
=====================

Synthesizes dubbed audio using voice cloning (Coqui XTTS v2).

Features:
- Voice cloning from speaker samples
- Speaker-consistent voices throughout video
- Duration alignment to match original timing
- Multi-language support

Models:
- XTTS v2: High quality voice cloning (24+ languages)
- Piper: Fast inference (fallback, less natural)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import time
import json
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tidaldub.workers.base import BaseWorker, WorkerConfig, ProcessingResult
from tidaldub.queues import Task, QueueName

# Lazy imports
torch = None
TTS = None


def lazy_import():
    """Lazy import heavy dependencies"""
    global torch, TTS
    if torch is None:
        import torch as _torch
        torch = _torch
    if TTS is None:
        from TTS.api import TTS as _TTS
        TTS = _TTS


class TTSWorker(BaseWorker):
    """
    Text-to-speech synthesis worker with voice cloning.
    
    Input: Translated transcript + speaker voice samples
    Output: Dubbed audio segments (per speaker, per language)
    
    Key features:
    - Uses original speaker voice to clone for target language
    - Maintains speaker identity across the video
    - Adjusts speed to match original segment duration
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.tts = None
        self.speaker_samples = {}  # Cache for speaker audio samples
        
        # Output directory
        self.data_dir = Path(self.global_config.get("paths", {}).get("data_dir", "./data"))
        
        # Get quality preset
        quality_preset = self.global_config.get("quality", {}).get("preset", "balanced")
        presets = self.global_config.get("quality", {}).get("presets", {})
        self.tts_model = presets.get(quality_preset, {}).get("tts_model", "xtts_v2")
    
    def initialize(self) -> None:
        """Load the TTS model"""
        lazy_import()
        
        print(f"[{self.worker_id}] Loading TTS model: {self.tts_model}")
        
        device = self.config.gpu_device if torch.cuda.is_available() else "cpu"
        print(f"[{self.worker_id}] Using device: {device}")
        
        if self.tts_model == "xtts_v2":
            # Coqui XTTS v2 - best quality voice cloning
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        elif self.tts_model == "piper":
            # Piper TTS - faster but less natural
            # Note: Piper uses different API, simplified here
            print(f"[{self.worker_id}] Using XTTS v2 as fallback (Piper integration pending)")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        else:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        self.device = device
        print(f"[{self.worker_id}] TTS model loaded successfully")
    
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Synthesize dubbed audio for translated segments.
        
        Payload:
            translated_transcript_path: Path to translated transcript
            speaker_map_path: Path to speaker map with voice samples
            vocals_path: Path to original vocals (for speaker samples)
            target_language: Target language code
            job_id: Job ID
        
        Returns:
            ProcessingResult with paths to synthesized audio segments
        """
        translated_path = Path(task.payload.get("translated_transcript_path"))
        speaker_map_path = Path(task.payload.get("speaker_map_path"))
        vocals_path = Path(task.payload.get("vocals_path"))
        target_lang = task.payload.get("target_language")
        job_id = task.job_id
        
        # Validate inputs
        if not translated_path.exists():
            return ProcessingResult(success=False, error=f"Translated transcript not found: {translated_path}")
        
        print(f"[{self.worker_id}] Synthesizing TTS for {target_lang}...")
        
        try:
            # Check for completed checkpoint
            progress = self.load_progress(task)
            completed_segments = set(progress.get("completed_segments", [])) if progress else set()
            
            start_time = time.time()
            
            # Load translated transcript
            with open(translated_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # Load speaker map
            if speaker_map_path.exists():
                with open(speaker_map_path, 'r', encoding='utf-8') as f:
                    speaker_map = json.load(f)
            else:
                speaker_map = {"speakers": ["SPEAKER_00"]}
            
            # Extract speaker voice samples
            if vocals_path.exists():
                self._prepare_speaker_samples(vocals_path, speaker_map, transcript)
            
            segments = transcript.get("segments", [])
            
            output_dir = self.data_dir / "temp" / job_id / "tts" / target_lang
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_paths = []
            
            for seg in segments:
                seg_id = str(seg.get("id", 0))
                
                if seg_id in completed_segments:
                    output_path = output_dir / f"segment_{int(seg_id):04d}.wav"
                    if output_path.exists():
                        output_paths.append(str(output_path))
                    continue
                
                text = seg.get("translated_text", seg.get("text", ""))
                speaker = seg.get("speaker", "SPEAKER_00")
                original_duration = seg.get("end", 0) - seg.get("start", 0)
                
                if not text.strip():
                    # Empty segment - create silence
                    output_path = self._create_silence(
                        output_dir / f"segment_{int(seg_id):04d}.wav",
                        original_duration,
                    )
                else:
                    # Synthesize speech
                    output_path = self._synthesize_segment(
                        text=text,
                        speaker=speaker,
                        target_lang=target_lang,
                        original_duration=original_duration,
                        output_path=output_dir / f"segment_{int(seg_id):04d}.wav",
                    )
                
                output_paths.append(str(output_path))
                completed_segments.add(seg_id)
                
                # Save progress periodically
                if int(seg_id) % 10 == 0:
                    self.save_progress(task, {
                        "completed_segments": list(completed_segments),
                        "target_language": target_lang,
                    })
                    print(f"[{self.worker_id}] Progress: {len(completed_segments)}/{len(segments)} segments")
            
            synthesis_time = time.time() - start_time
            
            # Concatenate all segments into single audio file
            final_audio_path = output_dir / f"dubbed_{target_lang}.wav"
            self._concatenate_segments(output_paths, transcript["segments"], final_audio_path)
            
            metrics = {
                "synthesis_time_sec": synthesis_time,
                "segments_synthesized": len(segments),
                "target_language": target_lang,
            }
            
            # Save completion checkpoint
            self.save_progress(task, {
                "completed": True,
                "completed_segments": list(completed_segments),
                "output_paths": [str(final_audio_path)],
                "metrics": metrics,
            })
            
            print(f"[{self.worker_id}] TTS completed in {synthesis_time:.1f}s")
            
            return ProcessingResult(
                success=True,
                output_paths=[str(final_audio_path)],
                metrics=metrics,
            )
            
        except Exception as e:
            import traceback
            return ProcessingResult(
                success=False,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )
    
    def _prepare_speaker_samples(
        self,
        vocals_path: Path,
        speaker_map: dict,
        transcript: dict,
    ) -> None:
        """Extract voice samples for each speaker for cloning"""
        import torchaudio
        
        print(f"[{self.worker_id}] Extracting speaker voice samples...")
        
        waveform, sample_rate = torchaudio.load(vocals_path)
        
        # Group segments by speaker
        speaker_segments = {}
        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "SPEAKER_00")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        # Extract samples for each speaker
        for speaker, segments in speaker_segments.items():
            if speaker in self.speaker_samples:
                continue  # Already cached
            
            # Collect audio from speaker's segments (up to 30 seconds)
            chunks = []
            total_duration = 0
            
            for seg in segments[:10]:  # Use first 10 segments max
                if total_duration >= 30:
                    break
                
                start_sample = int(seg.get("start", 0) * sample_rate)
                end_sample = int(seg.get("end", 0) * sample_rate)
                
                chunk = waveform[:, start_sample:end_sample]
                if chunk.shape[-1] > 0:
                    chunks.append(chunk)
                    total_duration += seg.get("end", 0) - seg.get("start", 0)
            
            if chunks:
                combined = torch.cat(chunks, dim=-1)
                
                # Save to temp file for XTTS
                sample_path = self.temp_dir / f"speaker_sample_{speaker}.wav"
                torchaudio.save(str(sample_path), combined, sample_rate)
                
                self.speaker_samples[speaker] = str(sample_path)
                print(f"[{self.worker_id}] Extracted sample for {speaker}: {total_duration:.1f}s")
    
    def _synthesize_segment(
        self,
        text: str,
        speaker: str,
        target_lang: str,
        original_duration: float,
        output_path: Path,
    ) -> Path:
        """Synthesize speech for a single segment"""
        import torchaudio
        
        # Get speaker sample
        speaker_wav = self.speaker_samples.get(speaker)
        
        if speaker_wav is None:
            # Use default voice if no sample available
            speaker_wav = self.speaker_samples.get(list(self.speaker_samples.keys())[0]) if self.speaker_samples else None
        
        # Map language codes for XTTS
        lang_map = {
            "en": "en", "es": "es", "fr": "fr", "de": "de",
            "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
            "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
            "zh": "zh-cn", "ja": "ja", "ko": "ko", "hi": "hi",
        }
        tts_lang = lang_map.get(target_lang, "en")
        
        try:
            if speaker_wav:
                # Voice cloning synthesis
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=tts_lang,
                )
            else:
                # Default voice (no cloning)
                wav = self.tts.tts(text=text, language=tts_lang)
            
            # Convert to tensor
            wav_tensor = torch.tensor(wav).unsqueeze(0)
            
            # Adjust duration to match original
            synthesized_duration = len(wav) / 22050  # XTTS default sample rate
            
            if synthesized_duration > 0 and abs(synthesized_duration - original_duration) > 0.1:
                wav_tensor = self._time_stretch(
                    wav_tensor,
                    synthesized_duration,
                    original_duration,
                    22050,
                )
            
            # Save audio
            torchaudio.save(str(output_path), wav_tensor, 22050)
            
            return output_path
            
        except Exception as e:
            print(f"[{self.worker_id}] TTS error for segment: {e}")
            # Create silence as fallback
            return self._create_silence(output_path, original_duration)
    
    def _time_stretch(
        self,
        audio: torch.Tensor,
        current_duration: float,
        target_duration: float,
        sample_rate: int,
    ) -> torch.Tensor:
        """Time-stretch audio to match target duration"""
        try:
            import librosa
            
            # Convert to numpy
            audio_np = audio.squeeze().numpy()
            
            # Calculate stretch factor
            stretch_factor = current_duration / target_duration
            
            # Clamp stretch factor to reasonable range
            stretch_factor = max(0.5, min(2.0, stretch_factor))
            
            # Time stretch using librosa
            stretched = librosa.effects.time_stretch(
                audio_np,
                rate=stretch_factor,
            )
            
            return torch.tensor(stretched).unsqueeze(0)
            
        except Exception as e:
            print(f"[{self.worker_id}] Time stretch failed: {e}")
            return audio
    
    def _create_silence(
        self,
        output_path: Path,
        duration: float,
        sample_rate: int = 22050,
    ) -> Path:
        """Create a silent audio file"""
        import torchaudio
        
        num_samples = int(duration * sample_rate)
        silence = torch.zeros(1, num_samples)
        
        torchaudio.save(str(output_path), silence, sample_rate)
        
        return output_path
    
    def _concatenate_segments(
        self,
        segment_paths: List[str],
        segments: List[dict],
        output_path: Path,
    ) -> None:
        """Concatenate segments with proper timing"""
        import torchaudio
        
        if not segment_paths:
            return
        
        sample_rate = 22050  # XTTS default
        
        # Calculate total duration
        total_duration = segments[-1].get("end", 0) if segments else 0
        total_samples = int(total_duration * sample_rate)
        
        # Create output buffer
        output = torch.zeros(1, total_samples)
        
        # Place each segment at correct position
        for i, (path, seg) in enumerate(zip(segment_paths, segments)):
            if not Path(path).exists():
                continue
            
            audio, sr = torchaudio.load(path)
            
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                audio = resampler(audio)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Calculate position
            start_sample = int(seg.get("start", 0) * sample_rate)
            end_sample = start_sample + audio.shape[-1]
            
            # Clamp to output bounds
            if end_sample > total_samples:
                audio = audio[:, :total_samples - start_sample]
                end_sample = total_samples
            
            # Mix into output (allows overlapping segments)
            if start_sample < total_samples:
                output[:, start_sample:end_sample] += audio
        
        # Normalize
        max_val = output.abs().max()
        if max_val > 1.0:
            output = output / max_val
        
        torchaudio.save(str(output_path), output, sample_rate)
        print(f"[{self.worker_id}] Saved concatenated audio: {output_path}")
    
    def cleanup(self) -> None:
        """Release resources"""
        self.speaker_samples.clear()
        
        if self.tts is not None:
            del self.tts
            self.tts = None
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[{self.worker_id}] Cleanup complete")


def create_tts_worker(global_config: dict):
    """Factory function to create TTS worker"""
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    from tidaldub.queues import QueueManager
    
    state_dir = global_config.get("paths", {}).get("state_dir", "./state")
    
    config = WorkerConfig(
        worker_type="tts",
        queue=QueueName.TTS,
        gpu_device=global_config.get("gpu", {}).get("device", "cuda:0"),
    )
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(global_config)
    
    return TTSWorker(
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
    
    worker = create_tts_worker(config)
    worker.run()
