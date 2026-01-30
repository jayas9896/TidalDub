# TidalDub Running Guide 🚀

Complete guide on how to run TidalDub, monitor progress, and troubleshoot issues.

---

## 📍 Quick Reference: File Locations

| Item | Location | Description |
|------|----------|-------------|
| **Input Videos** | `data/input/` | Place source videos here |
| **Output Videos** | `data/output/` | Final dubbed videos appear here |
| **Temp Files** | `data/temp/<job_id>/` | Intermediate processing files |
| **Job State** | `state/jobs/<job_id>/` | JSON state files for each job |
| **Logs** | `logs/tidaldub.log` | Application logs |
| **Event Log** | `logs/events.jsonl` | Structured event stream |
| **AI Models** | `models/` | Downloaded AI models (~20GB) |
| **Config** | `config.yaml` | Main configuration file |

---

## 🎬 Submitting a Job

### Basic Usage

```powershell
# 1. Place your video in the input folder
Copy-Item "C:\Videos\myMovie.mp4" "data\input\"

# 2. Submit the job
uv run tidaldub submit data/input/myMovie.mp4 --audio-langs es,fr,de

# 3. Note the job ID returned (e.g., job_a1b2c3d4)
```

### Command Options

```powershell
uv run tidaldub submit <video_path> [OPTIONS]

Options:
  --audio-langs TEXT      Languages for voice dubbing (comma-separated)
                          Example: es,fr,de,ja,ko
  
  --subtitle-langs TEXT   Languages for subtitles only (comma-separated)
                          Example: zh,ru,ar,vi
  
  --quality TEXT          Quality preset: fast|balanced|quality
                          Default: balanced
  
  --priority INTEGER      Job priority (higher = processed first)
                          Default: 0
```

### Examples

```powershell
# Dub into Spanish and French with Japanese subtitles
uv run tidaldub submit data/input/movie.mp4 --audio-langs es,fr --subtitle-langs ja

# Use fast preset for quicker processing (lower quality)
uv run tidaldub submit data/input/movie.mp4 --audio-langs es --quality fast

# High priority job (processed before others)
uv run tidaldub submit data/input/urgent.mp4 --audio-langs de --priority 10
```

---

## 📊 Monitoring Progress

### Check Job Status

```powershell
uv run tidaldub status <job_id>
```

**Example output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                    TidalDub Job Status                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Job ID:     job_a1b2c3d4                                        ║
║  Status:     RUNNING                                             ║
║  Video:      myMovie.mp4                                         ║
║  Languages:  es, fr, de                                          ║
║  Started:    2024-01-29 14:30:00                                 ║
║  Elapsed:    00:15:23                                            ║
╠══════════════════════════════════════════════════════════════════╣
║                         Stage Progress                           ║
╠══════════════════════════════════════════════════════════════════╣
║  ✅ Separation      [████████████████████] 100%  (2m 15s)        ║
║  ✅ Transcription   [████████████████████] 100%  (1m 45s)        ║
║  ✅ Diarization     [████████████████████] 100%  (1m 30s)        ║
║  🔄 Translation     [████████████░░░░░░░░]  65%  (running...)    ║
║     └─ Spanish:     [████████████████████] 100%                  ║
║     └─ French:      [██████████░░░░░░░░░░]  50%                  ║
║     └─ German:      [░░░░░░░░░░░░░░░░░░░░]   0%  (queued)        ║
║  ⏳ TTS             [░░░░░░░░░░░░░░░░░░░░]   0%  (pending)       ║
║  ⏳ Mixing          [░░░░░░░░░░░░░░░░░░░░]   0%  (pending)       ║
║  ⏳ Muxing          [░░░░░░░░░░░░░░░░░░░░]   0%  (pending)       ║
╠══════════════════════════════════════════════════════════════════╣
║  Overall Progress:  [██████████░░░░░░░░░░]  45%                  ║
║  ETA:               ~25 minutes remaining                        ║
╚══════════════════════════════════════════════════════════════════╝
```

### List All Jobs

```powershell
# List all jobs
uv run tidaldub list

# Filter by status
uv run tidaldub list --status running
uv run tidaldub list --status completed
uv run tidaldub list --status failed

# Show recent jobs only
uv run tidaldub list --limit 10
```

**Example output:**
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              TidalDub Jobs                                     ║
╠══════════════════╦═══════════════════════╦══════════╦═════════════╦═══════════╣
║  Job ID          ║  Video                ║  Status  ║  Progress   ║  Elapsed  ║
╠══════════════════╬═══════════════════════╬══════════╬═════════════╬═══════════╣
║  job_a1b2c3d4    ║  myMovie.mp4          ║  RUNNING ║  45%        ║  15m 23s  ║
║  job_e5f6g7h8    ║  documentary.mp4      ║  QUEUED  ║  0%         ║  --       ║
║  job_i9j0k1l2    ║  shortFilm.mp4        ║  DONE    ║  100%       ║  8m 45s   ║
║  job_m3n4o5p6    ║  tutorial.mp4         ║  FAILED  ║  65%        ║  12m 10s  ║
╚══════════════════╩═══════════════════════╩══════════╩═════════════╩═══════════╝
```

---

## 📁 Where to Find Files

### Input Video
```
data/input/myMovie.mp4          # Place your source video here
```

### Output Video (Final Result)
```
data/output/myMovie_dubbed.mkv  # Multi-track MKV with all languages
data/output/myMovie_es.mp4      # Single-language MP4 (if requested)
```

### Intermediate Files (Per Job)
```
data/temp/job_a1b2c3d4/
├── audio/
│   ├── original.wav            # Extracted original audio
│   ├── vocals.wav              # Separated vocals
│   └── accompaniment.wav       # Music + effects
├── transcription/
│   └── transcript.json         # Text with timestamps
├── diarization/
│   └── diarized.json           # Speaker-labeled segments
├── translation/
│   ├── es.json                 # Spanish translation
│   ├── fr.json                 # French translation
│   └── de.json                 # German translation
├── tts/
│   ├── es/
│   │   ├── segment_001.wav     # Spanish TTS segment 1
│   │   ├── segment_002.wav     # Spanish TTS segment 2
│   │   └── ...
│   ├── fr/
│   └── de/
├── mixed/
│   ├── es.wav                  # Final Spanish audio track
│   ├── fr.wav                  # Final French audio track
│   └── de.wav                  # Final German audio track
└── subtitles/
    ├── es.srt                  # Spanish subtitles
    ├── fr.srt                  # French subtitles
    └── de.srt                  # German subtitles
```

### State Files (Per Job)
```
state/jobs/job_a1b2c3d4/
├── job_state.json              # Overall job metadata
├── separation.json             # Separation stage state
├── transcription.json          # Transcription stage state
├── diarization.json            # Diarization stage state
├── translation_es.json         # Translation state (per language)
├── translation_fr.json
├── tts_es.json                 # TTS state (per language)
├── mixing_es.json              # Mixing state (per language)
└── muxing.json                 # Final muxing state
```

---

## 📈 Viewing Logs and Metrics

### Application Log
```powershell
# View live logs
Get-Content logs/tidaldub.log -Tail 50 -Wait

# Or use:
uv run tidaldub logs --follow
```

### Event Log (Structured)
```powershell
# View recent events
Get-Content logs/events.jsonl -Tail 20 | ConvertFrom-Json | Format-Table

# Example event:
# {
#   "timestamp": "2024-01-29T14:30:00.123Z",
#   "event": "stage_completed",
#   "job_id": "job_a1b2c3d4",
#   "stage": "separation",
#   "duration_sec": 135.5
# }
```

### Per-Stage Progress

View detailed progress for each stage:

```powershell
# View stage state file
Get-Content state/jobs/job_a1b2c3d4/translation_es.json | ConvertFrom-Json

# Output:
# {
#   "status": "completed",
#   "progress_percent": 100,
#   "started_at": "2024-01-29T14:35:00Z",
#   "completed_at": "2024-01-29T14:37:00Z",
#   "segments_total": 150,
#   "segments_completed": 150,
#   "error": null
# }
```

### GPU Metrics

Monitor GPU usage during processing:

```powershell
# Windows
nvidia-smi -l 1

# Or use:
nvidia-smi dmon -d 1
```

---

## 🔄 Managing Jobs

### Resume a Crashed Job

If TidalDub crashes or is interrupted:

```powershell
uv run tidaldub resume <job_id>
```

This will:
1. Detect where the job was interrupted
2. Reset running tasks to pending
3. Continue from the last checkpoint

### Cancel a Running Job

```powershell
uv run tidaldub cancel <job_id>
```

### Delete a Job

```powershell
# Delete job and all associated files
uv run tidaldub delete <job_id>

# Keep output, delete temp files only
uv run tidaldub delete <job_id> --keep-output
```

---

## 💀 Dead Letter Queue (DLQ) Management

Failed tasks are moved to the DLQ after max retries.

### List DLQ Items

```powershell
uv run tidaldub dlq list
```

**Example output:**
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          Dead Letter Queue                                     ║
╠═══════════════════╦═════════════════════╦════════════════════════════════════╣
║  DLQ ID           ║  Stage              ║  Error                             ║
╠═══════════════════╬═════════════════════╬════════════════════════════════════╣
║  dlq_x1y2z3       ║  translation_ar     ║  CUDA out of memory                ║
║  dlq_a4b5c6       ║  tts_ko             ║  Voice sample too short            ║
╚═══════════════════╩═════════════════════╩════════════════════════════════════╝
```

### Retry a Failed Task

```powershell
# Retry a specific DLQ item
uv run tidaldub dlq retry dlq_x1y2z3

# Retry all items
uv run tidaldub dlq retry-all
```

### View DLQ Item Details

```powershell
uv run tidaldub dlq inspect dlq_x1y2z3
```

### Delete DLQ Item

```powershell
uv run tidaldub dlq delete dlq_x1y2z3
```

---

## 🚨 Troubleshooting

### Problem: Job stuck at 0%

**Check if workers are running:**
```powershell
uv run tidaldub worker status
```

**Start workers manually:**
```powershell
# Start all workers
uv run tidaldub worker start

# Or start specific worker
uv run tidaldub worker start --stage separation
```

### Problem: CUDA out of memory

**Solutions:**

1. **Switch to fast preset:**
   ```yaml
   # config.yaml
   quality:
     preset: "fast"
   ```

2. **Enable CPU offload:**
   ```yaml
   # config.yaml
   hardware:
     use_cpu_offload: true
   ```

3. **Close other GPU applications**

4. **Check GPU memory:**
   ```powershell
   nvidia-smi
   ```

### Problem: Translation quality is poor

**Solutions:**

1. **Use quality preset:**
   ```yaml
   quality:
     preset: "quality"
   ```

2. **Verify source language detection:**
   Check `data/temp/<job_id>/transcription/transcript.json` for detected language.

3. **Add source language hint:**
   ```powershell
   uv run tidaldub submit video.mp4 --source-lang en --audio-langs es
   ```

### Problem: Voice cloning sounds robotic

**Solutions:**

1. **Check voice sample quality:**
   - Source audio should be clear
   - Minimal background noise
   - At least 10 seconds of speech per speaker

2. **Use quality preset:**
   ```yaml
   quality:
     preset: "quality"
   ```

### Problem: Audio/video sync issues

**Solutions:**

1. **Check source video:**
   ```powershell
   ffprobe data/input/video.mp4
   ```

2. **Verify audio extraction:**
   Check `data/temp/<job_id>/audio/original.wav` duration matches video.

3. **Manual resync (if needed):**
   ```powershell
   ffmpeg -i output.mkv -itsoffset 0.5 -i audio.wav -map 0:v -map 1:a -c copy fixed.mkv
   ```

### Problem: Job failed with unknown error

**Debug steps:**

1. **Check application log:**
   ```powershell
   Get-Content logs/tidaldub.log -Tail 100 | Select-String "ERROR"
   ```

2. **Check event log:**
   ```powershell
   Get-Content logs/events.jsonl -Tail 50
   ```

3. **Check stage state file:**
   ```powershell
   Get-Content state/jobs/<job_id>/<stage>.json
   ```

4. **Check DLQ for error details:**
   ```powershell
   uv run tidaldub dlq list
   uv run tidaldub dlq inspect <dlq_id>
   ```

---

## ⏱️ Performance Tips

### Maximize GPU Utilization

```yaml
# config.yaml
gpu:
  performance:
    torch_compile:
      enabled: true
      mode: "reduce-overhead"
    flash_attention:
      enabled: true
    tf32:
      enabled: true
```

### Speed Up Mixing (CPU-Bound)

```yaml
# config.yaml - Use more CPU cores
workers:
  mixing: 8  # Increase for more CPU cores

hardware:
  cpu_optimization:
    process_pool_workers: 8  # Match mixing workers
```

### Enable Pipeline Streaming

```yaml
# config.yaml - Start next stage at 50% completion
pipeline:
  streaming:
    enabled: true
    stream_threshold_percent: 50
```

### Use Redis for Faster Queue

```yaml
# config.yaml
queues:
  use_redis: true
  redis_url: "redis://localhost:6379/0"
  pubsub:
    enabled: true
```

---

## 📊 Processing Time Estimates

For a 1-hour video on RTX 5070 8GB:

| Stage | Estimated Time | Notes |
|-------|----------------|-------|
| Separation | 5-10 min | GPU-bound |
| Transcription | 3-5 min | GPU-bound |
| Diarization | 2-4 min | GPU-bound |
| Translation | 3-5 min per language | GPU-bound |
| TTS | 10-20 min per language | GPU-bound |
| Mixing | 1-2 min per language | CPU-bound (parallel) |
| Muxing | 1-2 min | CPU-bound |

**Total for 3 languages:** ~60-90 minutes (with pipeline streaming)

---

## ✅ Success Checklist

After a job completes, verify:

- [ ] Output file exists: `data/output/<video>_dubbed.mkv`
- [ ] File plays in VLC/MPC
- [ ] All language tracks are selectable
- [ ] Subtitles display correctly
- [ ] Audio is in sync with video
- [ ] Voice quality is acceptable

---

## 🎉 You're Ready!

You now know how to:
- Submit videos for dubbing
- Monitor progress in real-time
- Find all input/output files
- View logs and metrics
- Troubleshoot common issues

Happy dubbing! 🌊🎬
