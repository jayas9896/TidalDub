"""
Translation Worker
==================

Translates transcript segments to target languages.

Options:
1. SeamlessM4T (Meta) - High quality, GPU required, 100+ languages
2. Argos Translate - Lighter, works offline, fewer languages

Features:
- Batch translation for efficiency
- Preserves speaker IDs and timestamps
- Per-segment checkpointing for long videos
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
torch = None


def lazy_import():
    """Lazy import heavy dependencies"""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


# Language code mapping for SeamlessM4T
SEAMLESS_LANG_CODES = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "zh": "cmn",
    "ja": "jpn",
    "ko": "kor",
    "ar": "arb",
    "hi": "hin",
    "nl": "nld",
    "pl": "pol",
    "tr": "tur",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "uk": "ukr",
    "cs": "ces",
}


class TranslationWorker(BaseWorker):
    """
    Text translation worker.
    
    Input: Diarized transcript (source language)
    Output: Translated transcript for each target language
    
    Preserves:
    - Segment boundaries and timestamps
    - Speaker IDs
    - Word-level timing (approximated)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.translator = None
        self.processor = None
        
        # Output directory
        self.data_dir = Path(self.global_config.get("paths", {}).get("data_dir", "./data"))
        
        # Get target languages
        self.target_languages = self.global_config.get("languages", {}).get("audio", [])
    
    def initialize(self) -> None:
        """Load the translation model"""
        lazy_import()
        
        print(f"[{self.worker_id}] Loading SeamlessM4T translation model...")
        
        device = self.config.gpu_device if torch.cuda.is_available() else "cpu"
        print(f"[{self.worker_id}] Using device: {device}")
        
        try:
            from seamless_communication.inference import Translator
            
            self.translator = Translator(
                model_name_or_card="seamlessM4T_v2_large",
                vocoder_name_or_card="vocoder_v2",
                device=torch.device(device),
            )
            self.use_seamless = True
            
        except ImportError:
            print(f"[{self.worker_id}] SeamlessM4T not available, trying Argos...")
            try:
                import argostranslate.package
                import argostranslate.translate
                
                # Download/update language packages
                argostranslate.package.update_package_index()
                self.use_seamless = False
                self.argos = argostranslate
                
            except ImportError:
                raise ImportError(
                    "Neither seamless_communication nor argostranslate available. "
                    "Please install one of them."
                )
        
        self.device = device
        print(f"[{self.worker_id}] Translation model loaded successfully")
    
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Translate transcript to target languages.
        
        Payload:
            transcript_path: Path to diarized transcript
            target_languages: List of target language codes
            job_id: Job ID
        
        Returns:
            ProcessingResult with translated transcript paths
        """
        transcript_path = Path(task.payload.get("transcript_path"))
        target_languages = task.payload.get("target_languages", self.target_languages)
        job_id = task.job_id
        
        if not transcript_path.exists():
            return ProcessingResult(
                success=False,
                error=f"Transcript not found: {transcript_path}"
            )
        
        print(f"[{self.worker_id}] Translating to: {target_languages}")
        
        try:
            # Check for completed checkpoint
            progress = self.load_progress(task)
            completed_langs = set(progress.get("completed_languages", [])) if progress else set()
            
            start_time = time.time()
            
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            source_lang = transcript.get("language", "en")
            segments = transcript.get("segments", [])
            
            print(f"[{self.worker_id}] Source language: {source_lang}")
            print(f"[{self.worker_id}] Segments to translate: {len(segments)}")
            
            output_dir = self.data_dir / "temp" / job_id / "translation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_paths = []
            
            for target_lang in target_languages:
                if target_lang in completed_langs:
                    print(f"[{self.worker_id}] Skipping {target_lang} (already completed)")
                    output_path = output_dir / f"transcript_{target_lang}.json"
                    if output_path.exists():
                        output_paths.append(str(output_path))
                    continue
                
                if target_lang == source_lang:
                    print(f"[{self.worker_id}] Skipping {target_lang} (same as source)")
                    continue
                
                print(f"[{self.worker_id}] Translating to {target_lang}...")
                
                translated_segments = []
                
                for i, seg in enumerate(segments):
                    text = seg.get("text", "")
                    
                    if not text.strip():
                        translated_segments.append({
                            **seg,
                            "translated_text": "",
                        })
                        continue
                    
                    # Translate
                    translated_text = self._translate_text(
                        text,
                        source_lang,
                        target_lang,
                    )
                    
                    translated_segments.append({
                        **seg,
                        "original_text": text,
                        "translated_text": translated_text,
                    })
                    
                    # Save progress periodically
                    if i % 50 == 0:
                        self.save_progress(task, {
                            "completed_languages": list(completed_langs),
                            "current_language": target_lang,
                            "current_segment": i,
                        })
                
                # Create translated transcript
                translated_transcript = {
                    **transcript,
                    "target_language": target_lang,
                    "source_language": source_lang,
                    "segments": translated_segments,
                }
                
                # Save translated transcript
                output_path = output_dir / f"transcript_{target_lang}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(translated_transcript, f, indent=2, ensure_ascii=False)
                
                output_paths.append(str(output_path))
                completed_langs.add(target_lang)
                
                print(f"[{self.worker_id}] Saved: {output_path}")
                
                # Update checkpoint
                self.save_progress(task, {
                    "completed_languages": list(completed_langs),
                })
            
            translation_time = time.time() - start_time
            
            metrics = {
                "translation_time_sec": translation_time,
                "languages_translated": len(output_paths),
                "segments_per_language": len(segments),
            }
            
            # Mark as complete
            self.save_progress(task, {
                "completed": True,
                "completed_languages": list(completed_langs),
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
    
    def _translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text using available translator"""
        if self.use_seamless:
            return self._translate_seamless(text, source_lang, target_lang)
        else:
            return self._translate_argos(text, source_lang, target_lang)
    
    def _translate_seamless(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate using SeamlessM4T"""
        src_lang = SEAMLESS_LANG_CODES.get(source_lang, source_lang)
        tgt_lang = SEAMLESS_LANG_CODES.get(target_lang, target_lang)
        
        try:
            translated_text, _, _ = self.translator.predict(
                text,
                task_str="t2tt",  # Text-to-text translation
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            return str(translated_text)
        except Exception as e:
            print(f"[{self.worker_id}] Translation error: {e}")
            return text  # Return original on error
    
    def _translate_argos(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate using Argos Translate"""
        try:
            translated = self.argos.translate.translate(
                text,
                source_lang,
                target_lang,
            )
            return translated
        except Exception as e:
            print(f"[{self.worker_id}] Translation error: {e}")
            return text
    
    def cleanup(self) -> None:
        """Release resources"""
        if self.translator is not None:
            del self.translator
            self.translator = None
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[{self.worker_id}] Cleanup complete")


def create_translation_worker(global_config: dict):
    """Factory function to create translation worker"""
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    from tidaldub.queues import QueueManager
    
    state_dir = global_config.get("paths", {}).get("state_dir", "./state")
    
    config = WorkerConfig(
        worker_type="translation",
        queue=QueueName.TRANSLATION,
        gpu_device=global_config.get("gpu", {}).get("device", "cuda:0"),
    )
    
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(global_config)
    
    return TranslationWorker(
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
    
    worker = create_translation_worker(config)
    worker.run()
