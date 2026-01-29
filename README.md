# TidalDub 🌊🎬

**Fully local video dubbing pipeline with enterprise-grade reliability.**

TidalDub automatically dubs video content into multiple languages using:
- 🎤 Voice cloning (preserves original speaker voices)
- 🔊 Professional audio separation & mixing
- 📝 Multi-language subtitles
- 💾 Crash-proof state management (File FSM + SQLite)

## Features

### Audio Processing
- **Demucs** - Separates vocals, music, drums, bass, and sound effects
- **Whisper** - State-of-the-art speech-to-text with word timestamps
- **pyannote-audio** - Speaker diarization (who said what)

### Translation & Synthesis
- **SeamlessM4T** - Meta's multilingual translation (100+ languages)
- **Coqui XTTS v2** - Voice cloning TTS (maintains speaker identity)
- **Duration alignment** - Dubbed audio matches original timing

### Professional Output
- Multi-track MKV with selectable audio/subtitle languages
- Broadcast-standard loudness normalization (-16 LUFS)
- Optional web UI for monitoring

### Reliability
- **File-based FSM** - Ground truth state survives anything
- **SQLite cache** - Fast queries, auto-rebuilt if corrupted
- **Dead Letter Queue** - Failed tasks don't block the pipeline
- **Checkpoint recovery** - Resume from exact crash point

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- FFmpeg installed and in PATH
- ~20GB for AI models

### Quick Start

```bash
# Clone the repository
cd tidal-whirlpool

# Create main venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install orchestrator
pip install -e .

# Create worker venvs (each has its own dependencies)
cd workers/separation && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ../transcription && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ../diarization && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ../translation && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ../tts && python -m venv venv && venv\Scripts\pip install -r requirements.txt
cd ../mixing && python -m venv venv && venv\Scripts\pip install -r requirements.txt
```

## Usage

### Submit a Video

```bash
tidaldub submit movie.mp4 --audio-langs es,fr,de --subtitle-langs es,fr,de,ja
```

### Check Status

```bash
tidaldub status job_abc123def456
```

### List All Jobs

```bash
tidaldub list
```

### Resume Crashed Job

```bash
tidaldub resume job_abc123def456
```

### Manage Dead Letter Queue

```bash
tidaldub dlq list
tidaldub dlq retry dlq_item_id
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               TidalDub                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Orchestrator ──► File FSM (ground truth) ──► SQLite (queries) ──► Redis (opt) │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │Separation│─►│Transcribe│─►│Diarize │─►│Translate│─►│  TTS   │─►│  Mix    │  │
│  │ Demucs  │  │ Whisper │  │pyannote │  │SeamlessM4T│ │Coqui   │  │ FFmpeg  │  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Target languages
languages:
  audio: [es, fr, de, ja, ko]
  subtitles: [es, fr, de, ja, ko, zh, ru]

# Quality preset (fast / balanced / quality)
quality:
  preset: balanced
```

## Project Structure

```
tidal-whirlpool/
├── config.yaml              # Global configuration
├── tidaldub/                # Main package
│   ├── cli.py               # Command-line interface
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── recovery.py          # Crash recovery
│   ├── muxer.py             # Video muxing
│   ├── state/               # Reliability layer
│   │   ├── fsm.py           # File-based FSM
│   │   ├── database.py      # SQLite state
│   │   └── events.py        # Event logging
│   ├── queues/              # Queue infrastructure
│   │   └── manager.py       # Queue manager
│   └── workers/             # Worker framework
│       └── base.py          # Base worker class
├── workers/                 # Isolated worker venvs
│   ├── separation/          # Demucs
│   ├── transcription/       # Whisper
│   ├── diarization/         # pyannote
│   ├── translation/         # SeamlessM4T
│   ├── tts/                 # Coqui XTTS
│   └── mixing/              # Audio mixing
├── state/                   # Runtime state (FSM files)
├── data/                    # Processing data
│   ├── input/               # Source videos
│   ├── temp/                # Intermediate files
│   └── output/              # Final outputs
└── models/                  # Downloaded AI models
```

## License

MIT License - See LICENSE file for details.
