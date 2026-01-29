"""
Parallel Mixing Worker
======================

High-performance audio mixing with parallel language processing.

Optimized for Intel Core Ultra 9 275HX (24 cores):
- Process 4 languages simultaneously
- Uses ProcessPoolExecutor for CPU-bound mixing
- Non-blocking async interface
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def mix_audio_process(args: dict) -> dict:
    """
    CPU-bound mixing function that runs in a separate process.
    
    This is the function that gets parallelized across CPU cores.
    """
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    
    dubbed_path = Path(args["dubbed_path"])
    background_path = Path(args["background_path"])
    output_path = Path(args["output_path"])
    target_lang = args["target_lang"]
    
    try:
        # Load audio files
        dubbed_sr, dubbed = wavfile.read(dubbed_path)
        bg_sr, background = wavfile.read(background_path)
        
        # Convert to float
        dubbed = dubbed.astype(np.float32) / 32768.0
        background = background.astype(np.float32) / 32768.0
        
        # Ensure mono for dubbed
        if dubbed.ndim > 1:
            dubbed = dubbed.mean(axis=1)
        
        # Ensure stereo for background
        if background.ndim == 1:
            background = np.stack([background, background], axis=1)
        
        # Resample if needed
        if dubbed_sr != bg_sr:
            dubbed = signal.resample(
                dubbed,
                int(len(dubbed) * bg_sr / dubbed_sr)
            )
            dubbed_sr = bg_sr
        
        # Match lengths
        min_len = min(len(dubbed), len(background))
        dubbed = dubbed[:min_len]
        background = background[:min_len]
        
        # Apply vocal EQ
        # High-pass filter (remove low rumble)
        b, a = signal.butter(2, 80 / (dubbed_sr / 2), btype='highpass')
        dubbed = signal.filtfilt(b, a, dubbed)
        
        # Low-pass filter (remove harshness)
        b, a = signal.butter(2, 12000 / (dubbed_sr / 2), btype='lowpass')
        dubbed = signal.filtfilt(b, a, dubbed)
        
        # Presence boost (2-5kHz)
        b, a = signal.butter(2, [2000 / (dubbed_sr / 2), 5000 / (dubbed_sr / 2)], btype='bandpass')
        presence = signal.filtfilt(b, a, dubbed)
        dubbed = dubbed + presence * 0.2
        
        # Add room reverb
        delay_samples = int(dubbed_sr * 0.02)
        reverb = np.zeros_like(dubbed)
        for i in range(1, 5):
            delay = delay_samples * i
            gain = 0.3 ** i
            if len(dubbed) > delay:
                reverb[delay:] += dubbed[:-delay] * gain
        dubbed = dubbed + reverb * 0.15
        
        # Convert to stereo
        dubbed_stereo = np.stack([dubbed * 0.7, dubbed * 0.7], axis=1)
        
        # Compression
        envelope = np.abs(dubbed_stereo).max(axis=1)
        envelope = uniform_filter1d(envelope, size=int(len(envelope) * 0.01))
        
        threshold_lin = 10 ** (-20 / 20)
        gain = np.ones_like(envelope)
        above_threshold = envelope > threshold_lin
        
        if above_threshold.any():
            db_above = 20 * np.log10(envelope[above_threshold] / threshold_lin)
            db_reduction = db_above * (1 - 1/4)
            gain[above_threshold] = 10 ** (-db_reduction / 20)
        
        dubbed_stereo = dubbed_stereo * gain.reshape(-1, 1)
        
        # Mix
        bg_level = 0.5  # -6 dB
        mixed = dubbed_stereo + background * bg_level
        
        # Limiter
        mixed = np.clip(mixed, -1.0, 1.0)
        
        # Loudness normalization
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(bg_sr)
            current_loudness = meter.integrated_loudness(mixed)
            if current_loudness > -70:
                mixed = pyln.normalize.loudness(mixed, current_loudness, -16)
        except Exception:
            peak = np.abs(mixed).max()
            if peak > 0:
                mixed = mixed / peak * 0.9
        
        # Save
        mixed_int = (mixed * 32767).astype(np.int16)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(output_path), bg_sr, mixed_int)
        
        return {
            "success": True,
            "output_path": str(output_path),
            "target_lang": target_lang,
            "duration_sec": len(mixed) / bg_sr,
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "target_lang": target_lang,
        }


class ParallelMixingWorker:
    """
    Parallel audio mixing worker.
    
    Processes multiple languages simultaneously using ProcessPoolExecutor.
    Optimized for Intel Core Ultra 9 275HX with 24 cores.
    """
    
    def __init__(self, config: dict, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
        self._executor: Optional[ProcessPoolExecutor] = None
    
    def initialize(self):
        """Initialize the process pool"""
        # Use spawn to avoid CUDA issues in child processes
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx,
        )
        print(f"[ParallelMixing] Initialized with {self.max_workers} workers")
    
    def process_all_languages(
        self,
        job_id: str,
        background_path: Path,
        languages: List[str],
    ) -> Dict[str, dict]:
        """
        Process all languages in parallel.
        
        Returns dict mapping language code to result.
        """
        if not self._executor:
            self.initialize()
        
        start_time = time.time()
        
        # Build tasks for each language
        futures = {}
        for lang in languages:
            dubbed_path = self.data_dir / "temp" / job_id / "tts" / lang / f"dubbed_{lang}.wav"
            output_path = self.data_dir / "temp" / job_id / "mixed" / f"mixed_{lang}.wav"
            
            if not dubbed_path.exists():
                print(f"[ParallelMixing] Skipping {lang} - dubbed audio not found")
                continue
            
            args = {
                "dubbed_path": str(dubbed_path),
                "background_path": str(background_path),
                "output_path": str(output_path),
                "target_lang": lang,
            }
            
            future = self._executor.submit(mix_audio_process, args)
            futures[future] = lang
        
        # Collect results as they complete
        results = {}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                result = future.result()
                results[lang] = result
                
                if result["success"]:
                    print(f"[ParallelMixing] ✓ {lang} completed")
                else:
                    print(f"[ParallelMixing] ✗ {lang} failed: {result.get('error')}")
                    
            except Exception as e:
                results[lang] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"[ParallelMixing] ✗ {lang} exception: {e}")
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results.values() if r.get("success"))
        
        print(f"[ParallelMixing] Completed {successful}/{len(languages)} languages in {total_time:.1f}s")
        print(f"[ParallelMixing] Parallel speedup: {len(languages) / max(total_time, 0.1):.1f}x")
        
        return results
    
    async def process_all_languages_async(
        self,
        job_id: str,
        background_path: Path,
        languages: List[str],
    ) -> Dict[str, dict]:
        """Async wrapper for parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_all_languages,
            job_id,
            background_path,
            languages,
        )
    
    def cleanup(self):
        """Shutdown the executor"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        print("[ParallelMixing] Cleanup complete")


def create_parallel_mixing_worker(config: dict) -> ParallelMixingWorker:
    """Factory function to create parallel mixing worker"""
    max_workers = config.get("workers", {}).get("mixing", 4)
    
    # Limit to available CPU cores (leave some for system)
    cpu_count = os.cpu_count() or 4
    max_workers = min(max_workers, max(1, cpu_count // 3))
    
    return ParallelMixingWorker(config, max_workers=max_workers)


if __name__ == "__main__":
    """Test parallel mixing"""
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    worker = create_parallel_mixing_worker(config)
    worker.initialize()
    
    # Test would require actual audio files
    print("ParallelMixingWorker ready for testing")
    
    worker.cleanup()
