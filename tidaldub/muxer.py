"""
Final Video Muxer
=================

Combines original video with:
- Multiple dubbed audio tracks
- Multiple subtitle tracks
- Using FFmpeg for professional output
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json


class VideoMuxer:
    """
    Muxes video with multiple audio and subtitle tracks.
    
    Output format: MKV (best container for multiple tracks)
    
    Command structure:
    ffmpeg -i video.mp4 \
           -i dubbed_es.wav -i dubbed_fr.wav ... \
           -i subtitles_es.srt -i subtitles_fr.srt ... \
           -map 0:v -map 0:a -map 1:a -map 2:a ... \
           -c:v copy -c:a aac \
           -metadata:s:a:0 language=eng -metadata:s:a:0 title="Original" \
           -metadata:s:a:1 language=spa -metadata:s:a:1 title="Spanish" \
           output.mkv
    """
    
    # Language code to FFmpeg metadata
    LANG_NAMES = {
        "en": ("eng", "English"),
        "es": ("spa", "Spanish"),
        "fr": ("fra", "French"),
        "de": ("deu", "German"),
        "it": ("ita", "Italian"),
        "pt": ("por", "Portuguese"),
        "ru": ("rus", "Russian"),
        "zh": ("chi", "Chinese"),
        "ja": ("jpn", "Japanese"),
        "ko": ("kor", "Korean"),
        "ar": ("ara", "Arabic"),
        "hi": ("hin", "Hindi"),
        "nl": ("nld", "Dutch"),
        "pl": ("pol", "Polish"),
        "tr": ("tur", "Turkish"),
        "vi": ("vie", "Vietnamese"),
    }
    
    def __init__(self, data_dir: str | Path, output_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def mux(
        self,
        job_id: str,
        video_path: str | Path,
        audio_tracks: Dict[str, str],  # {lang_code: audio_path}
        subtitle_tracks: Dict[str, str],  # {lang_code: subtitle_path}
        include_original: bool = True,
    ) -> Path:
        """
        Mux video with multiple audio and subtitle tracks.
        
        Args:
            job_id: Job identifier
            video_path: Path to original video
            audio_tracks: Dict mapping language code to audio file path
            subtitle_tracks: Dict mapping language code to subtitle file path
            include_original: Whether to include original audio as first track
        
        Returns:
            Path to output video file
        """
        video_path = Path(video_path)
        output_path = self.output_dir / f"{job_id}_dubbed.mkv"
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite
        
        # Input: original video
        cmd.extend(["-i", str(video_path)])
        
        # Input: dubbed audio tracks
        audio_inputs = []
        for lang, audio_path in sorted(audio_tracks.items()):
            if Path(audio_path).exists():
                cmd.extend(["-i", str(audio_path)])
                audio_inputs.append(lang)
        
        # Input: subtitle tracks
        subtitle_inputs = []
        for lang, sub_path in sorted(subtitle_tracks.items()):
            if Path(sub_path).exists():
                cmd.extend(["-i", str(sub_path)])
                subtitle_inputs.append(lang)
        
        # Mapping
        # Video from input 0
        cmd.extend(["-map", "0:v"])
        
        # Original audio (if requested)
        audio_idx = 0
        if include_original:
            cmd.extend(["-map", "0:a"])
            audio_idx += 1
        
        # Dubbed audio tracks
        for i, lang in enumerate(audio_inputs, start=1):
            cmd.extend(["-map", f"{i}:a"])
        
        # Subtitle tracks
        sub_start_idx = 1 + len(audio_inputs)
        for i, lang in enumerate(subtitle_inputs):
            cmd.extend(["-map", f"{sub_start_idx + i}:0"])
        
        # Codecs
        cmd.extend(["-c:v", "copy"])  # Copy video (no re-encoding)
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])  # AAC audio
        cmd.extend(["-c:s", "copy"])  # Copy subtitles
        
        # Metadata for original audio
        if include_original:
            cmd.extend([
                f"-metadata:s:a:0", "language=eng",
                f"-metadata:s:a:0", "title=Original",
            ])
        
        # Metadata for dubbed audio tracks
        for i, lang in enumerate(audio_inputs):
            track_idx = i + (1 if include_original else 0)
            ffmpeg_lang, lang_name = self.LANG_NAMES.get(lang, (lang, lang.upper()))
            cmd.extend([
                f"-metadata:s:a:{track_idx}", f"language={ffmpeg_lang}",
                f"-metadata:s:a:{track_idx}", f"title={lang_name} (Dubbed)",
            ])
        
        # Metadata for subtitle tracks
        for i, lang in enumerate(subtitle_inputs):
            ffmpeg_lang, lang_name = self.LANG_NAMES.get(lang, (lang, lang.upper()))
            cmd.extend([
                f"-metadata:s:s:{i}", f"language={ffmpeg_lang}",
                f"-metadata:s:s:{i}", f"title={lang_name}",
            ])
        
        # Set default tracks
        if include_original:
            cmd.extend(["-disposition:a:0", "default"])
        
        # Output
        cmd.append(str(output_path))
        
        print(f"Running FFmpeg command:")
        print(" ".join(cmd[:20]) + " ...")
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"Output saved to: {output_path}")
        
        return output_path
    
    def generate_subtitles(
        self,
        job_id: str,
        translations_dir: Path,
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Generate SRT subtitle files from translations.
        
        Returns dict mapping language code to subtitle file path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        subtitle_files = {}
        
        for trans_file in translations_dir.glob("transcript_*.json"):
            # Extract language from filename
            lang = trans_file.stem.replace("transcript_", "")
            
            with open(trans_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # Generate SRT
            srt_path = output_dir / f"subtitles_{lang}.srt"
            self._write_srt(transcript, srt_path)
            
            subtitle_files[lang] = str(srt_path)
            print(f"Generated subtitles: {srt_path}")
        
        return subtitle_files
    
    def _write_srt(self, transcript: dict, output_path: Path) -> None:
        """Write transcript to SRT format"""
        segments = transcript.get("segments", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, start=1):
                start = self._format_srt_time(seg.get("start", 0))
                end = self._format_srt_time(seg.get("end", 0))
                text = seg.get("translated_text", seg.get("text", ""))
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_muxer(config: dict) -> VideoMuxer:
    """Create muxer from config"""
    data_dir = config.get("paths", {}).get("data_dir", "./data")
    output_dir = config.get("paths", {}).get("output_dir", "./data/output")
    
    return VideoMuxer(data_dir, output_dir)
