"""
Audio Source Separation Worker
==============================

Separates audio into stems using Demucs (Meta's AI model):
- vocals: Dialogue to be replaced with dubbed audio
- drums/bass/other: Music components (preserved)
- Stems are recombined minus vocals + new dubbed vocals

Models available:
- htdemucs: Default hybrid transformer model
- htdemucs_ft: Fine-tuned version (higher quality, slower)
- mdx_extra: MDX-Net architecture
"""

import os
import sys
from pathlib import Path
from typing import Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tidaldub.workers.base import BaseWorker, WorkerConfig, ProcessingResult
from tidaldub.queues import Task, QueueName

# Lazy imports for heavy dependencies
torch = None
demucs = None


def lazy_import():
    """Lazy import heavy dependencies"""
    global torch, demucs
    if torch is None:
        import torch as _torch
        torch = _torch
    if demucs is None:
        import demucs as _demucs
        demucs = _demucs


class SeparationWorker(BaseWorker):
    """
    Audio source separation worker using Demucs.
    
    Input: Original video/audio file
    Output: Separated stems (vocals, music, sfx, ambience)
    
    The vocals stem will be discarded and replaced with dubbed audio.
    All other stems are preserved and remixed with the dubbed vocals.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = None
        self.model_name = self.global_config.get("quality", {}).get(
            "presets", {}
        ).get(
            self.global_config.get("quality", {}).get("preset", "balanced"), {}
        ).get("separation_model", "htdemucs")
        
        # Output paths will be relative to job data directory
        self.data_dir = Path(self.global_config.get("paths", {}).get("data_dir", "./data"))
    
    def initialize(self) -> None:
        """Load the Demucs model"""
        lazy_import()
        
        print(f"[{self.worker_id}] Loading Demucs model: {self.model_name}")
        
        # Import Demucs components
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        self.apply_model = apply_model
        
        # Load model
        device = self.config.gpu_device if torch.cuda.is_available() else "cpu"
        print(f"[{self.worker_id}] Using device: {device}")
        
        self.model = get_model(self.model_name)
        self.model.to(device)
        self.model.eval()
        
        self.device = device
        
        print(f"[{self.worker_id}] Demucs model loaded successfully")
    
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Separate audio from video file into stems.
        
        Payload:
            audio_path: Path to extracted audio file
            job_id: Job ID for output organization
        
        Returns:
            ProcessingResult with paths to separated stems
        """
        audio_path = Path(task.payload.get("audio_path"))
        job_id = task.job_id
        
        if not audio_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Audio file not found: {audio_path}"
            )
        
        print(f"[{self.worker_id}] Processing: {audio_path}")
        
        try:
            # Resume from checkpoint if available
            progress = self.load_progress(task)
            if progress and progress.get("stems_completed"):
                print(f"[{self.worker_id}] Resuming from checkpoint")
                return ProcessingResult(
                    success=True,
                    output_paths=progress.get("output_paths", []),
                    metrics=progress.get("metrics", {}),
                )
            
            import torchaudio
            
            # Load audio
            print(f"[{self.worker_id}] Loading audio file...")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed (Demucs expects 44100 Hz)
            if sample_rate != 44100:
                print(f"[{self.worker_id}] Resampling from {sample_rate} to 44100 Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, 44100)
                waveform = resampler(waveform)
                sample_rate = 44100
            
            # Ensure stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            
            # Add batch dimension
            waveform = waveform.unsqueeze(0).to(self.device)
            
            # Process through model
            print(f"[{self.worker_id}] Running separation model...")
            start_time = time.time()
            
            with torch.no_grad():
                sources = self.apply_model(
                    self.model,
                    waveform,
                    device=self.device,
                    progress=True,
                    num_workers=0,
                )
            
            separation_time = time.time() - start_time
            print(f"[{self.worker_id}] Separation completed in {separation_time:.1f}s")
            
            # Save separated stems
            output_dir = self.data_dir / "temp" / job_id / "separation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Demucs htdemucs has 4 sources: drums, bass, other, vocals
            source_names = self.model.sources  # ['drums', 'bass', 'other', 'vocals']
            
            output_paths = []
            for idx, name in enumerate(source_names):
                stem = sources[0, idx].cpu()
                stem_path = output_dir / f"{name}.wav"
                
                torchaudio.save(
                    str(stem_path),
                    stem,
                    sample_rate,
                    encoding="PCM_F",
                    bits_per_sample=32,
                )
                
                output_paths.append(str(stem_path))
                print(f"[{self.worker_id}] Saved: {stem_path}")
            
            # Create "background" mix (everything except vocals)
            background = sources[0, :3].sum(dim=0).cpu()  # drums + bass + other
            background_path = output_dir / "background.wav"
            
            torchaudio.save(
                str(background_path),
                background,
                sample_rate,
                encoding="PCM_F",
                bits_per_sample=32,
            )
            output_paths.append(str(background_path))
            print(f"[{self.worker_id}] Saved background mix: {background_path}")
            
            # Calculate metrics
            metrics = {
                "separation_time_sec": separation_time,
                "audio_duration_sec": waveform.shape[-1] / sample_rate,
                "sample_rate": sample_rate,
                "num_stems": len(source_names),
            }
            
            # Save checkpoint
            self.save_progress(task, {
                "stems_completed": True,
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


def create_separation_worker(global_config: dict):
    """Factory function to create separation worker"""
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    from tidaldub.queues import QueueManager
    
    state_dir = global_config.get("paths", {}).get("state_dir", "./state")
    
    config = WorkerConfig(
        worker_type="separation",
        queue=QueueName.SEPARATION,
        gpu_device=global_config.get("gpu", {}).get("device", "cuda:0"),
    )
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(global_config)
    
    return SeparationWorker(
        config=config,
        fsm=fsm,
        database=database,
        queue_manager=queue_manager,
        global_config=global_config,
    )


if __name__ == "__main__":
    """Run as standalone worker"""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create and run worker
    worker = create_separation_worker(config)
    worker.run()
