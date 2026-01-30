# TidalDub Architecture 🏗️

This document provides a comprehensive overview of TidalDub's architecture, including component interactions, data flow, and design decisions.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    TidalDub                                          │
│                                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐              │
│  │   CLI Interface  │───▶│   Orchestrator   │◀──▶│  Recovery System │              │
│  │    (cli.py)      │    │ (orchestrator.py)│    │  (recovery.py)   │              │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────┘              │
│                                   │                                                  │
│           ┌───────────────────────┼───────────────────────┐                         │
│           ▼                       ▼                       ▼                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐              │
│  │   File-based FSM │◀──▶│  SQLite Database │◀──▶│   Queue Manager  │              │
│  │     (fsm.py)     │    │   (database.py)  │    │   (manager.py)   │              │
│  │  [Ground Truth]  │    │  [Fast Queries]  │    │ [Task Dispatch]  │              │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘              │
│                                                           │                         │
│                                                           ▼                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐    │
│  │                           AI Pipeline Workers                               │    │
│  │                                                                            │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │    │
│  │  │Separation│──▶│Transcribe│──▶│ Diarize  │──▶│Translate │──▶│   TTS    │ │    │
│  │  │  Demucs  │   │ Whisper  │   │ pyannote │   │SeamlessM4T│   │Coqui XTTS│ │    │
│  │  │  ~5GB    │   │  ~3GB    │   │  ~3GB    │   │  ~5GB    │   │  ~5GB    │ │    │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │    │
│  │                                                                            │    │
│  │                              ┌──────────┐                                  │    │
│  │                              │  Mixing  │◀─────────────────────────────────│    │
│  │                              │  FFmpeg  │   (Parallel CPU Processing)      │    │
│  │                              │  0GB GPU │                                  │    │
│  │                              └──────────┘                                  │    │
│  └────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                           │
│                                         ▼                                           │
│                              ┌──────────────────┐                                   │
│                              │      Muxer       │                                   │
│                              │    (muxer.py)    │                                   │
│                              │  Multi-track MKV │                                   │
│                              └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
                         ┌─────────────────────────────────────┐
                         │         INPUT VIDEO (MP4)           │
                         │      data/input/myVideo.mp4         │
                         └──────────────┬──────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: SEPARATION (Demucs)                                                        │
│ ─────────────────────────────                                                       │
│ Input:  Original video audio                                                        │
│ Output: vocals.wav, accompaniment.wav (music + effects)                             │
│ VRAM:   ~5GB                                                                        │
│ Time:   ~2-5 min per hour of video                                                 │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: TRANSCRIPTION (faster-whisper)                                              │
│ ─────────────────────────────────────────                                           │
│ Input:  vocals.wav                                                                  │
│ Output: transcript.json (text + word-level timestamps)                              │
│ VRAM:   ~3GB (faster-whisper saves 50% vs original Whisper)                        │
│ Time:   ~1-3 min per hour of video                                                 │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DIARIZATION (pyannote-audio)                                               │
│ ─────────────────────────────────────────                                           │
│ Input:  vocals.wav + transcript.json                                                │
│ Output: diarized.json (speaker labels for each segment)                             │
│ VRAM:   ~3GB                                                                        │
│ Time:   ~1-2 min per hour of video                                                 │
│ Note:   Requires HuggingFace token for model access                                │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: TRANSLATION (SeamlessM4T)                                                  │
│ ─────────────────────────────────────                                               │
│ Input:  diarized.json + source language                                             │
│ Output: translations/{lang}.json for each target language                           │
│ VRAM:   ~5GB                                                                        │
│ Time:   ~1-3 min per language per hour of video                                    │
│ Note:   Supports 100+ languages                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: TTS (Coqui XTTS v2)                                                        │
│ ───────────────────────────────                                                     │
│ Input:  translations/{lang}.json + speaker voice samples                            │
│ Output: tts/{lang}/segment_*.wav (synthesized speech per segment)                   │
│ VRAM:   ~5GB                                                                        │
│ Time:   ~5-15 min per language per hour of video                                   │
│ Note:   Clones original speaker voices to target language                          │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: MIXING (FFmpeg + Python)                                              ⚡   │
│ ─────────────────────────────────────                                               │
│ Input:  tts/{lang}/*.wav + accompaniment.wav + original audio                      │
│ Output: mixed/{lang}.wav (professional dubbed audio track)                          │
│ VRAM:   0GB (CPU-only)                                                              │
│ CPU:    4 parallel workers (uses all 24 cores)                                     │
│ Time:   ~30 sec per language (parallel processing!)                                │
│                                                                                     │
│ Audio Processing Chain:                                                            │
│   1. Align TTS segments to original timing                                         │
│   2. Apply EQ to match original voice characteristics                              │
│   3. Add reverb to match room acoustics                                            │
│   4. Apply compression for consistent levels                                       │
│   5. Mix with music + effects at proper ratios                                     │
│   6. Normalize to -16 LUFS (broadcast standard)                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: MUXING (FFmpeg)                                                            │
│ ────────────────────────────                                                        │
│ Input:  original video + all mixed/{lang}.wav + subtitle files                      │
│ Output: data/output/myVideo_dubbed.mkv                                              │
│                                                                                     │
│ Output Tracks:                                                                      │
│   • Video: Original video stream (passthrough)                                     │
│   • Audio: Original + all dubbed language tracks (selectable)                      │
│   • Subtitles: SRT/WebVTT for each language (selectable)                           │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │     OUTPUT VIDEO (MKV)              │
                         │  data/output/myVideo_dubbed.mkv     │
                         │                                     │
                         │  ✓ Original video preserved         │
                         │  ✓ 10+ audio tracks (selectable)    │
                         │  ✓ 15+ subtitle tracks              │
                         └─────────────────────────────────────┘
```

---

## 🧩 Component Details

### 1. CLI Interface (`tidaldub/cli.py`)

Entry point for all user interactions.

```
Commands:
  submit    Submit a video for dubbing
  status    Check job progress  
  list      List all jobs
  resume    Resume a crashed job
  dlq       Manage Dead Letter Queue
  worker    Start worker processes
```

| Command | Example | Description |
|---------|---------|-------------|
| `submit` | `tidaldub submit video.mp4 --audio-langs es,fr` | Create new dubbing job |
| `status` | `tidaldub status job_abc123` | Show detailed progress |
| `list` | `tidaldub list --status running` | Filter jobs by status |
| `resume` | `tidaldub resume job_abc123` | Recover crashed job |
| `dlq list` | `tidaldub dlq list` | Show failed tasks |
| `dlq retry` | `tidaldub dlq retry item_id` | Retry failed task |

---

### 2. Orchestrator (`tidaldub/orchestrator.py`)

Central coordinator managing job lifecycle and pipeline stages.

**Responsibilities:**
- Job creation and validation
- Stage sequencing and dependency management
- Worker spawning and monitoring
- Progress tracking and status updates
- Error handling and retry logic
- Pipeline streaming (starting next stage early)

**Key Features:**

```python
# Pipeline streaming - start next stage at 50% completion
if progress >= stream_threshold:
    if not self._stage_already_queued(job_id, next_stage):
        self._queue_stage(job_id, next_stage)
```

---

### 3. State Management Layer

Three-tier reliability system ensuring data integrity:

#### File-based FSM (`state/fsm.py`) - Ground Truth
```
state/
├── jobs/
│   └── job_abc123/
│       ├── job_state.json      # Job metadata
│       ├── separation.json     # Stage state
│       ├── transcription.json
│       └── ...
└── lock files (cross-platform)
```

**Features:**
- Atomic writes (write to temp, then rename)
- Cross-platform file locking
- Human-readable JSON
- Survives any crash

#### SQLite Database (`state/database.py`) - Fast Queries
```sql
-- Indexed for fast lookups
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_segments_job ON segments(job_id);
```

**Features:**
- WAL mode for concurrent reads
- Index-optimized queries
- Auto-rebuild from FSM if corrupted

#### Event Log (`state/events.py`) - Audit Trail
```
logs/events.jsonl
{"ts": "2024-01-29T12:00:00", "event": "job_created", "job_id": "abc123"}
{"ts": "2024-01-29T12:00:05", "event": "stage_started", "stage": "separation"}
...
```

---

### 4. Queue Manager (`queues/manager.py`)

Handles task distribution with two backends:

#### Redis Backend (Primary)
```yaml
# config.yaml
queues:
  use_redis: true
  redis_url: "redis://localhost:6379/0"
  pubsub:
    enabled: true
    channel_prefix: "tidaldub:notify"
```

**Features:**
- Instant pub/sub notifications
- Zero polling latency
- Distributed worker support

#### SQLite Backend (Fallback)
```yaml
queues:
  use_redis: false  # Falls back to SQLite
  sqlite_wal_mode: true
```

**Features:**
- No external dependencies
- WAL mode for performance
- Automatic failover

#### Dead Letter Queue (DLQ)

Failed tasks are moved to DLQ after max retries:

```
┌────────────────────────────────────────────────┐
│                Queue Flow                       │
│                                                 │
│  Task ──▶ Queue ──▶ Worker ──▶ ✓ Complete     │
│                         │                       │
│                         ▼ (failure)             │
│                    Retry Queue                  │
│                         │                       │
│                         ▼ (max retries)         │
│                       DLQ                       │
│                         │                       │
│                         ▼ (manual)              │
│                    Retry/Delete                 │
└────────────────────────────────────────────────┘
```

---

### 5. AI Pipeline Workers

Each worker is an isolated Python package with its own dependencies:

| Worker | Model | VRAM | Purpose |
|--------|-------|------|---------|
| **Separation** | Demucs HT-Demucs | ~5GB | Isolate vocals from music/effects |
| **Transcription** | faster-whisper large-v3 | ~3GB | Speech-to-text with timestamps |
| **Diarization** | pyannote-audio | ~3GB | Speaker identification |
| **Translation** | SeamlessM4T large | ~5GB | Multi-language translation |
| **TTS** | Coqui XTTS v2 | ~5GB | Voice cloning speech synthesis |
| **Mixing** | FFmpeg/scipy | 0GB | Professional audio mixing |

**GPU Optimization (enabled for all GPU workers):**

```python
# From tidaldub/workers/base.py

# torch.compile for 2-3x faster inference
if torch_compile_enabled:
    model = torch.compile(model, mode="reduce-overhead")

# Flash Attention 2 for 50% less memory
if flash_attention_enabled:
    model = model.to_bettertransformer()

# TF32 for faster matrix operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

### 6. Muxer (`tidaldub/muxer.py`)

Final assembly of all components into output video.

**FFmpeg command structure:**
```bash
ffmpeg -i original.mp4 \
  -i mixed_es.wav -i mixed_fr.wav -i mixed_de.wav \
  -i subtitles_es.srt -i subtitles_fr.srt \
  -map 0:v -map 0:a -map 1:a -map 2:a -map 3:a \
  -map 4:s -map 5:s \
  -c:v copy -c:a aac -c:s srt \
  -metadata:s:a:0 language=eng -metadata:s:a:0 title="English (Original)" \
  -metadata:s:a:1 language=spa -metadata:s:a:1 title="Spanish" \
  output_dubbed.mkv
```

---

## ⚡ Performance Architecture

### Sequential GPU Processing

Workers run sequentially to stay within 8GB VRAM:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Separation│───▶│Transcribe│───▶│ Diarize │───▶│Translate│───▶│   TTS   │
│  ~5GB   │    │   ~3GB  │    │  ~3GB   │    │  ~5GB   │    │  ~5GB   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
   Clear          Clear          Clear          Clear          Clear
   VRAM           VRAM           VRAM           VRAM           VRAM
```

### Parallel CPU Processing (Mixing)

Mixing uses all 24 CPU cores:

```
                    ┌─────────────────┐
                    │  Mixing Stage   │
                    │ ProcessPoolExecutor(4) │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │  Worker 1  │    │  Worker 2  │    │  Worker 3  │    ...
    │  Spanish   │    │  French    │    │  German    │
    │  6 cores   │    │  6 cores   │    │  6 cores   │
    └────────────┘    └────────────┘    └────────────┘
```

### Pipeline Streaming

Next stage starts before current stage completes:

```
Time ─────────────────────────────────────────────────────────▶

Separation:    [████████████████████████████████████100%]
Transcription:              [██████████████████████████100%]
Diarization:                          [████████████████100%]
Translation:                                   [███████100%]
TTS:                                                [██100%]
Mixing:                                               [100%]

                    ↑ Starts at 50%
```

---

## 🔒 Reliability Design

### Three-Layer State Management

```
┌─────────────────────────────────────────────────────────────┐
│                    Reliability Layers                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: File FSM (Ground Truth)                           │
│  ├── Atomic writes prevent corruption                       │
│  ├── File locks prevent race conditions                     │
│  └── JSON is human-readable for debugging                   │
│                                                              │
│  Layer 2: SQLite Cache (Fast Queries)                       │
│  ├── O(1) lookups via indexes                              │
│  ├── WAL mode for concurrent reads                          │
│  └── Auto-rebuilds from FSM if corrupted                    │
│                                                              │
│  Layer 3: Queue System (Task Dispatch)                      │
│  ├── DLQ captures all failures                              │
│  ├── Exponential backoff for retries                        │
│  └── Redis pub/sub for instant notifications                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Crash Recovery Flow

```
┌────────────────────────────────────────────────────────────────┐
│                     Recovery Process                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Detect Unclean Shutdown                                    │
│     └── Check for stale PID files                              │
│                                                                 │
│  2. Load State from FSM                                        │
│     └── Read all job_state.json files                          │
│                                                                 │
│  3. Identify Interrupted Work                                  │
│     └── Find jobs/segments with status="running"               │
│                                                                 │
│  4. Reset to Last Checkpoint                                   │
│     └── Mark interrupted items as "pending"                    │
│                                                                 │
│  5. Requeue Tasks                                              │
│     └── Push pending work back to queue                        │
│                                                                 │
│  6. Resume Normal Operation                                    │
│     └── Workers pick up requeued tasks                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Architecture

```yaml
# config.yaml structure

paths:           # Directory locations
quality:         # Model selection presets
workers:         # Worker counts and async settings
queues:          # Redis/SQLite and DLQ settings
logging:         # Log levels and file settings
gpu:             # VRAM limits and optimizations
hardware:        # CPU/GPU specific tuning
pipeline:        # Streaming and parallelism
languages:       # Target audio/subtitle languages
```

---

## 📊 Metrics and Monitoring

TidalDub provides several monitoring points:

| Metric Location | Information |
|-----------------|-------------|
| `tidaldub status <job_id>` | Per-stage progress percentage |
| `state/jobs/<job_id>/` | Detailed state JSON files |
| `logs/tidaldub.log` | Full application logs |
| `logs/events.jsonl` | Structured event stream |
| Redis pub/sub | Real-time task notifications |

---

## 🏁 Summary

TidalDub's architecture prioritizes:

1. **Reliability** - Three-layer state management ensures no data loss
2. **Performance** - GPU optimization and parallel CPU processing
3. **Scalability** - Queue-based architecture supports distributed workers
4. **Observability** - Comprehensive logging and status tracking
5. **Maintainability** - Isolated workers with independent dependencies
