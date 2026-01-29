# TidalDub Quick Setup Guide
# ==========================

# Prerequisites:
# - Python 3.12, 3.13, or 3.14 (winget install Python.Python.3.13)
# - uv package manager (pip install uv OR winget install astral-sh.uv)
# - FFmpeg (winget install ffmpeg)
# - CUDA Toolkit 13.1 (winget install Nvidia.CUDA)
# - HuggingFace account (for pyannote models)

# Your Hardware:
# - MSI Crosshair 18 HX AI
# - Intel Core Ultra 9 275HX
# - NVIDIA RTX 5070 (8GB GDDR7)
# - 32GB DDR5 RAM

# ==========================================
# STEP 1: Install Prerequisites
# ==========================================

# Install Python 3.13 (latest stable)
winget install Python.Python.3.13

# Install uv (modern Rust-based package manager)
winget install astral-sh.uv

# Install FFmpeg (for audio/video processing)
winget install ffmpeg

# Install CUDA Toolkit 13.1 (matches PyTorch builds)
# Download from: https://developer.nvidia.com/cuda-downloads
# Or via winget:
winget install Nvidia.CUDA

# ==========================================
# STEP 2: Clone and Setup the Project
# ==========================================

cd tidal-whirlpool

# Sync all workspace packages (this resolves all dependencies)
# uv uses a lockfile (uv.lock) for deterministic builds
uv sync --all-packages

# ==========================================
# STEP 3: Install Worker-Specific Dependencies
# ==========================================

# Each worker has its own isolated environment via the workspace
# The main sync command handles everything, but for individual workers:

# For individual worker development/testing:
cd workers/separation && uv sync && cd ../..
cd workers/transcription && uv sync && cd ../..
cd workers/diarization && uv sync && cd ../..
cd workers/translation && uv sync && cd ../..
cd workers/tts && uv sync && cd ../..
cd workers/mixing && uv sync && cd ../..

# ==========================================
# STEP 4: Set up HuggingFace Token (for pyannote)
# ==========================================

# Create a token at: https://huggingface.co/settings/tokens
# Accept the pyannote model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

# Then set the environment variable:
# PowerShell (temporary):
$env:HUGGINGFACE_TOKEN = "your_token_here"

# PowerShell (permanent - add to profile):
[Environment]::SetEnvironmentVariable("HUGGINGFACE_TOKEN", "your_token", "User")

# ==========================================
# STEP 5: Verify CUDA is Working
# ==========================================

uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# PyTorch: 2.10.x+cu130
# CUDA available: True
# CUDA version: 13.0
# GPU: NVIDIA GeForce RTX 5070

# ==========================================
# STEP 6: Run TidalDub
# ==========================================

# Submit a video for dubbing:
uv run tidaldub submit video.mp4 --audio-langs es,fr,de

# Check status:
uv run tidaldub status job_abc123

# List all jobs:
uv run tidaldub list

# ==========================================
# HARDWARE OPTIMIZATION NOTES (RTX 5070 8GB)
# ==========================================

# Your RTX 5070 has 8GB GDDR7 VRAM. Here's how each worker uses it:
#
# Worker              VRAM Usage    Notes
# ----------------    ----------    -------------------------
# Separation          ~4-6 GB       Demucs htdemucs model
# Transcription       ~3 GB         faster-whisper large-v3
# Diarization         ~2-3 GB       pyannote-audio
# Translation         ~4-5 GB       SeamlessM4T large
# TTS                 ~4-5 GB       XTTS v2 voice cloning
# Mixing              0 GB          CPU-only (FFmpeg)
#
# Workers run sequentially, so you won't exceed 8GB.
# If you get OOM errors, switch to "fast" preset in config.yaml.

# ==========================================
# TROUBLESHOOTING
# ==========================================

# Out of Memory (OOM) errors:
# - Edit config.yaml, change preset from "balanced" to "fast"
# - This uses smaller models: whisper small (~2GB), translation medium (~3GB)

# CUDA not found:
# - Verify NVIDIA driver is 560+ (for RTX 5070)
# - Reinstall CUDA Toolkit 13.1
# - Restart PowerShell after installation

# PyTorch not using GPU:
# - Check torch.cuda.is_available()
# - Ensure CUDA Toolkit version matches PyTorch build (13.0)

# Dependency conflicts:
# - Delete uv.lock and .venv, then run: uv sync --all-packages
# - uv's resolver is much better than pip's

# Model download slow:
# - Models are cached in ./models directory
# - First run downloads ~20GB of AI models
# - Subsequent runs use cached models
