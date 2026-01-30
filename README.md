# TidalDub 🌊🎬

**Enterprise-grade, fully local video dubbing pipeline with AI-powered voice cloning.**

Transform any video into multiple languages while preserving original speaker voices, emotions, and timing. Built for reliability with crash-proof state management and optimized for NVIDIA RTX GPUs.

---

## 🎯 What TidalDub Does

TidalDub takes a video file and automatically:

1. **Extracts and separates audio** → Isolates vocals from music, effects, and background
2. **Transcribes speech** → Converts speech to text with precise word timestamps
3. **Identifies speakers** → Determines who said what (speaker diarization)
4. **Translates content** → Converts text to 100+ target languages
5. **Clones voices** → Synthesizes speech in target languages using original speaker voices
6. **Mixes professionally** → Combines dubbed audio with original background and music
7. **Outputs MKV** → Creates final video with selectable audio tracks and subtitles

**Result:** A professional-quality dubbed video with multi-language audio tracks and subtitles, all processed locally on your machine.

---

## ✨ Key Features

| Category | Features |
|----------|----------|
| **Audio AI** | Demucs source separation, faster-whisper transcription, pyannote speaker diarization |
| **Translation** | SeamlessM4T (100+ languages), preserves context and nuance |
| **Voice Cloning** | Coqui XTTS v2, maintains speaker identity across languages |
| **Audio Quality** | Professional mixing with EQ, reverb, compression, -16 LUFS loudness |
| **Output** | Multi-track MKV, selectable audio/subtitle tracks, WebVTT/SRT subtitles |
| **Reliability** | File-based FSM, SQLite caching, Dead Letter Queue, crash recovery |
| **Performance** | torch.compile, Flash Attention 2, parallel mixing, pipeline streaming |

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA with 6GB VRAM | RTX 3060+ with 8GB+ VRAM |
| **CPU** | 8 cores | 12+ cores (for parallel mixing) |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB free | 100GB+ (for models and temp files) |
| **OS** | Windows 10/11, Linux | Windows 11, Ubuntu 22.04+ |

> **Optimized for:** MSI Crosshair 18 HX (Intel Core Ultra 9 275HX, NVIDIA RTX 5070 8GB, 32GB RAM)

---

## 📁 Project Structure

```
TidalDub/
├── config.yaml              # Main configuration file
├── pyproject.toml           # Python dependencies (uv/pip)
├── uv.toml                  # uv workspace configuration
│
├── tidaldub/                # Core package
│   ├── cli.py               # Command-line interface
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── recovery.py          # Crash recovery system
│   ├── muxer.py             # Video/audio muxing
│   ├── async_worker.py      # Async worker framework
│   ├── state/               # State management
│   │   ├── fsm.py           # File-based state machine
│   │   ├── database.py      # SQLite cache
│   │   └── events.py        # Event logging
│   ├── queues/              # Queue infrastructure
│   │   └── manager.py       # Redis/SQLite queue manager
│   └── workers/             # Worker base classes
│       └── base.py          # GPU-optimized base worker
│
├── workers/                 # AI Pipeline Workers
│   ├── separation/          # Demucs audio separation
│   ├── transcription/       # faster-whisper STT
│   ├── diarization/         # pyannote speaker ID
│   ├── translation/         # SeamlessM4T translation
│   ├── tts/                 # Coqui XTTS voice cloning
│   └── mixing/              # Professional audio mixing
│
├── data/                    # Runtime data (created on first run)
│   ├── input/               # Place source videos here
│   ├── temp/                # Intermediate processing files
│   └── output/              # Final dubbed videos
│
├── state/                   # Job state files (FSM)
├── logs/                    # Application logs
└── models/                  # Downloaded AI models (~20GB)
```

---

## 🚀 Quick Start

### 1. Install Prerequisites

```powershell
# Windows (PowerShell as Admin)
winget install Python.Python.3.13
winget install astral-sh.uv
winget install Gyan.FFmpeg
# Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
```

### 2. Setup Project

```powershell
cd TidalDub
uv sync --all-packages
```

### 3. Set HuggingFace Token (for pyannote)

```powershell
# Get token from: https://huggingface.co/settings/tokens
# Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
$env:HUGGINGFACE_TOKEN = "your_token_here"
```

### 4. Run TidalDub

```powershell
# Place your video in data/input/
uv run tidaldub submit data/input/myVideo.mp4 --audio-langs es,fr,de

# Check progress
uv run tidaldub status <job_id>
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](./SETUP.md) | Detailed installation and configuration guide |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture and component details |
| [RUNNING.md](./RUNNING.md) | How to run jobs, monitor progress, troubleshoot |

---

## 🎯 Supported Languages

**Audio Dubbing (10 languages):**
Spanish, French, German, Portuguese, Italian, Japanese, Korean, Chinese, Hindi, Arabic

**Subtitles (15 languages):**
All audio languages plus Russian, Dutch, Polish, Turkish, Vietnamese

> SeamlessM4T supports 100+ languages. Edit `config.yaml` to add more.

---

## ⚡ Performance Optimizations

TidalDub is optimized for maximum performance:

- **torch.compile** with `reduce-overhead` mode → 2-3x faster inference
- **Flash Attention 2** → 50% less VRAM usage
- **Parallel Mixing** → 4 concurrent workers for CPU tasks
- **Redis Pub/Sub** → Instant task notifications (no polling)
- **Pipeline Streaming** → Next stage starts at 50% completion
- **CUDA Graphs** → Reduced kernel launch overhead

---

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
# Quality preset (fast/balanced/quality)
quality:
  preset: "balanced"

# Target languages
languages:
  audio: [es, fr, de, ja, ko]
  subtitles: [es, fr, de, ja, ko, zh, ru]

# Parallel CPU workers
workers:
  mixing: 4
```

---

## 📜 License

MIT License - See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <b>TidalDub</b> - Professional video dubbing, fully local, powered by AI
</p>
