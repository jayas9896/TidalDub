# TidalDub Setup Guide 🛠️

Complete installation and configuration guide for TidalDub.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Windows 10/11 or Linux (Ubuntu 22.04+)
- [ ] NVIDIA GPU with 8GB+ VRAM (RTX 3060 or newer recommended)
- [ ] 32GB RAM (16GB minimum)
- [ ] 100GB free disk space (for models and temp files)
- [ ] Internet connection (for initial model downloads)

---

## 🔧 Step 1: Install System Dependencies

### Windows (PowerShell as Administrator)

```powershell
# 1. Install Python 3.13
winget install Python.Python.3.13

# 2. Install uv (fast Python package manager)
winget install astral-sh.uv

# 3. Install FFmpeg (audio/video processing)
winget install Gyan.FFmpeg

# 4. Install Redis (optional, for faster queue processing)
# Download from: https://github.com/tporadowski/redis/releases
# Or use Docker: docker run -d -p 6379:6379 redis

# 5. Verify installations
python --version   # Should show 3.13.x
uv --version       # Should show 0.5.x or newer
ffmpeg -version    # Should show ffmpeg info
```

### Linux (Ubuntu/Debian)

```bash
# 1. Install Python 3.13
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install FFmpeg
sudo apt install ffmpeg

# 4. Install Redis (optional)
sudo apt install redis-server
sudo systemctl enable redis-server
```

---

## 🎮 Step 2: Install NVIDIA CUDA Toolkit

TidalDub requires CUDA 12.4+ for GPU acceleration.

### Windows

1. **Check your NVIDIA driver version:**
   ```powershell
   nvidia-smi
   ```
   You need driver version **545.0+** for CUDA 12.4

2. **Download CUDA Toolkit:**
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 → exe (network)
   - Download and run the installer

3. **Verify CUDA installation:**
   ```powershell
   nvcc --version  # Should show CUDA 12.4+
   ```

### Linux

```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

---

## 📦 Step 3: Install TidalDub

```powershell
# Navigate to project directory
cd TidalDub

# Install all packages using uv workspace
uv sync --all-packages

# This will:
# - Create a virtual environment in .venv/
# - Install all dependencies including PyTorch with CUDA
# - Set up all worker packages
```

### Verify PyTorch CUDA Support

```powershell
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.5.1+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 5070
```

---

## 🔑 Step 4: Configure HuggingFace Token

The diarization worker requires a HuggingFace token to download the pyannote-audio model.

### 1. Create HuggingFace Account
Go to: https://huggingface.co/join

### 2. Accept Model Terms
Visit and accept the terms at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Generate Access Token
Go to: https://huggingface.co/settings/tokens
- Click "New token"
- Name: `TidalDub`
- Type: `Read`
- Copy the token

### 4. Set Environment Variable

**Windows (PowerShell - Temporary):**
```powershell
$env:HUGGINGFACE_TOKEN = "hf_your_token_here"
```

**Windows (Permanent):**
```powershell
[Environment]::SetEnvironmentVariable("HUGGINGFACE_TOKEN", "hf_your_token_here", "User")
# Restart PowerShell after this
```

**Linux:**
```bash
echo 'export HUGGINGFACE_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## ⚙️ Step 5: Configure TidalDub

Edit `config.yaml` to customize settings:

### Quality Presets

```yaml
quality:
  preset: "balanced"  # Options: fast, balanced, quality
```

| Preset | VRAM Usage | Speed | Quality |
|--------|------------|-------|---------|
| `fast` | ~4GB | Fastest | Good |
| `balanced` | ~5GB | Medium | Very Good |
| `quality` | ~6GB | Slowest | Best |

### Target Languages

```yaml
languages:
  audio:       # Languages to dub (voice synthesis)
    - es       # Spanish
    - fr       # French
    - de       # German
  subtitles:   # Languages for subtitles only
    - ja       # Japanese
    - ko       # Korean
    - zh       # Chinese
```

### Redis Configuration (Optional)

If you installed Redis:
```yaml
queues:
  use_redis: true
  redis_url: "redis://localhost:6379/0"
```

If not using Redis, set:
```yaml
queues:
  use_redis: false  # Falls back to SQLite (still fast)
```

---

## 🗂️ Step 6: Create Directory Structure

TidalDub will create these automatically, but you can create them manually:

```powershell
# Windows
mkdir data\input
mkdir data\output
mkdir data\temp
mkdir state
mkdir logs
mkdir models
```

```bash
# Linux
mkdir -p data/{input,output,temp} state logs models
```

---

## ✅ Step 7: Verify Installation

Run the verification command:

```powershell
uv run tidaldub --help
```

**Expected output:**
```
Usage: tidaldub [OPTIONS] COMMAND [ARGS]...

  TidalDub - Professional video dubbing pipeline

Options:
  --config PATH  Path to config file
  --help         Show this message and exit.

Commands:
  dlq     Manage Dead Letter Queue
  list    List all jobs
  resume  Resume a crashed job
  status  Check job status
  submit  Submit a video for dubbing
  worker  Start worker processes
```

---

## 📊 Hardware-Specific Notes

### RTX 5070 (8GB VRAM)

Your configuration is optimized for this GPU:
- Use `balanced` preset (default)
- Workers run sequentially to fit in 8GB
- Parallel mixing uses 4 CPU workers

### RTX 3090/4090 (24GB VRAM)

You can increase throughput:
```yaml
quality:
  preset: "quality"  # Use highest quality models

workers:
  transcription: 2   # Run 2 instances
  translation: 2     # Run 2 instances
```

### Low VRAM (6GB or less)

Use the fast preset:
```yaml
quality:
  preset: "fast"     # Smallest models

hardware:
  use_cpu_offload: true  # Offload to CPU when needed
```

---

## 🚨 Common Installation Issues

### Issue: `torch.cuda.is_available()` returns False

**Solutions:**
1. Ensure NVIDIA driver is installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```powershell
   uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
   ```
3. Restart your terminal/PowerShell

### Issue: `uv sync` fails with dependency conflicts

**Solution:**
```powershell
# Clear cache and reinstall
Remove-Item -Recurse -Force .venv
Remove-Item uv.lock
uv sync --all-packages
```

### Issue: FFmpeg not found

**Solution:**
```powershell
# Windows - reinstall and ensure PATH is set
winget install Gyan.FFmpeg
# Close and reopen PowerShell
```

### Issue: CUDA out of memory

**Solutions:**
1. Switch to `fast` preset in config.yaml
2. Close other GPU applications
3. Enable CPU offload:
   ```yaml
   hardware:
     use_cpu_offload: true
   ```

### Issue: pyannote model download fails

**Solutions:**
1. Verify HuggingFace token is set: `echo $env:HUGGINGFACE_TOKEN`
2. Ensure you accepted model terms (links in Step 4)
3. Try manually downloading:
   ```powershell
   uv run python -c "from huggingface_hub import login; login()"
   # Enter your token when prompted
   ```

---

## 🎉 Installation Complete!

You're ready to start dubbing videos. See [RUNNING.md](./RUNNING.md) for usage instructions.

**Quick test:**
```powershell
# Place a short test video in data/input/
uv run tidaldub submit data/input/test.mp4 --audio-langs es
uv run tidaldub status <job_id>
```
