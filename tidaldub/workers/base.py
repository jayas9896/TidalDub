"""
Base Worker Framework
=====================

Provides the foundation for all pipeline workers (separation, transcription, etc.)

Features:
- Checkpoint loading/saving for crash recovery
- Retry with exponential backoff
- State updates to FSM → SQLite → Queues
- Graceful shutdown handling
- GPU memory management
- torch.compile integration for faster inference
- Flash Attention 2 support
"""

import os
import sys
import time
import signal
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass
import uuid
import socket

# Add parent to path for imports when running as standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tidaldub.state import AtomicFileState, Status
from tidaldub.state.database import StateDatabase
from tidaldub.state.events import JobEventLog, EventType
from tidaldub.queues import QueueManager, Task, QueueName


# =============================================================================
# Performance Optimization Helpers
# =============================================================================

def setup_gpu_optimizations(config: dict) -> None:
    """
    Set up GPU optimizations for faster inference.
    
    Call this at the start of GPU workers for maximum performance.
    """
    try:
        import torch
        
        gpu_config = config.get("gpu", {}).get("performance", {})
        
        # TF32 for faster matrix operations (RTX 30/40/50 series)
        if gpu_config.get("tf32", {}).get("enabled", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[GPU] TF32 enabled for faster matmul")
        
        # Flash Attention 2
        if gpu_config.get("flash_attention", {}).get("enabled", True):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("[GPU] Flash Attention 2 enabled")
        
        # cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        
        # Memory allocator optimization
        alloc_conf = config.get("gpu", {}).get("pytorch_cuda_alloc_conf")
        if alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
            print(f"[GPU] CUDA allocator config: {alloc_conf}")
            
    except ImportError:
        print("[GPU] PyTorch not available, skipping GPU optimizations")
    except Exception as e:
        print(f"[GPU] Optimization setup error: {e}")


def compile_model(model: Any, config: dict, model_name: str = "model") -> Any:
    """
    Apply torch.compile to a model for faster inference.
    
    Args:
        model: PyTorch model to compile
        config: Global config dict
        model_name: Name for logging
    
    Returns:
        Compiled model (or original if compilation fails/disabled)
    """
    try:
        import torch
        
        if not hasattr(torch, 'compile'):
            print(f"[torch.compile] Not available (requires PyTorch 2.0+)")
            return model
        
        gpu_config = config.get("gpu", {}).get("performance", {})
        compile_config = gpu_config.get("torch_compile", {})
        
        if not compile_config.get("enabled", True):
            print(f"[torch.compile] Disabled in config")
            return model
        
        mode = compile_config.get("mode", "reduce-overhead")
        
        print(f"[torch.compile] Compiling {model_name} with mode={mode}...")
        start_time = time.time()
        
        compiled = torch.compile(model, mode=mode)
        
        compile_time = time.time() - start_time
        print(f"[torch.compile] {model_name} compiled in {compile_time:.1f}s")
        
        return compiled
        
    except Exception as e:
        print(f"[torch.compile] Failed for {model_name}: {e}")
        return model


def clear_gpu_cache() -> None:
    """Clear GPU memory cache"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class WorkerConfig:
    """Configuration for a worker"""
    worker_type: str  # separation, transcription, etc.
    queue: QueueName
    
    # Worker identity
    worker_id: str = ""
    
    # Processing settings
    gpu_device: str = "cuda:0"
    batch_size: int = 1
    
    # Timeout settings
    task_timeout_sec: int = 3600  # 1 hour default
    heartbeat_interval_sec: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base_sec: int = 10
    
    def __post_init__(self):
        if not self.worker_id:
            hostname = socket.gethostname()
            self.worker_id = f"{self.worker_type}_{hostname}_{uuid.uuid4().hex[:8]}"


@dataclass
class ProcessingResult:
    """Result from processing a task"""
    success: bool
    output_paths: list = None
    metrics: dict = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def __post_init__(self):
        if self.output_paths is None:
            self.output_paths = []
        if self.metrics is None:
            self.metrics = {}


class BaseWorker(ABC):
    """
    Base class for all pipeline workers.
    
    Subclasses implement:
    - initialize(): Load models, set up resources
    - process_task(): Do the actual work
    - cleanup(): Release resources
    
    The base class handles:
    - Queue management
    - State updates (FSM, SQLite)
    - Checkpointing
    - Error handling and retries
    - Graceful shutdown
    """
    
    def __init__(
        self,
        config: WorkerConfig,
        fsm: AtomicFileState,
        database: StateDatabase,
        queue_manager: QueueManager,
        global_config: dict,
    ):
        self.config = config
        self.fsm = fsm
        self.database = database
        self.queue_manager = queue_manager
        self.global_config = global_config
        
        self.worker_id = config.worker_id
        self.running = False
        self.current_task: Optional[Task] = None
        
        # Paths
        self.state_dir = Path(global_config.get("paths", {}).get("state_dir", "./state"))
        self.temp_dir = Path(global_config.get("paths", {}).get("temp_dir", "./data/temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    # =========================================================================
    # Abstract Methods - Subclasses Must Implement
    # =========================================================================
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the worker: load models, set up GPU, etc.
        Called once when worker starts.
        """
        pass
    
    @abstractmethod
    def process_task(self, task: Task) -> ProcessingResult:
        """
        Process a single task.
        
        Args:
            task: The task to process
            
        Returns:
            ProcessingResult with success status and outputs
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources, unload models, etc.
        Called when worker shuts down.
        """
        pass
    
    # =========================================================================
    # Main Worker Loop
    # =========================================================================
    
    def run(self) -> None:
        """Main worker loop"""
        print(f"[{self.worker_id}] Starting worker...")
        
        try:
            self.initialize()
            self.running = True
            
            print(f"[{self.worker_id}] Worker initialized, listening on {self.config.queue.value}")
            
            while self.running:
                # Get next task from queue
                task = self.queue_manager.dequeue(
                    self.config.queue,
                    self.worker_id,
                    timeout=30,
                )
                
                if task is None:
                    continue
                
                self.current_task = task
                
                try:
                    self._process_with_recovery(task)
                except Exception as e:
                    print(f"[{self.worker_id}] Unhandled error: {e}")
                    traceback.print_exc()
                finally:
                    self.current_task = None
                    
        except Exception as e:
            print(f"[{self.worker_id}] Fatal error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
            print(f"[{self.worker_id}] Worker stopped")
    
    def _process_with_recovery(self, task: Task) -> None:
        """Process a task with checkpoint recovery"""
        job_id = task.job_id
        segment_id = task.payload.get("segment_id")
        stage = self.config.worker_type
        
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(task)
        if checkpoint and checkpoint.get("completed"):
            print(f"[{self.worker_id}] Task {task.id} already completed, skipping")
            self.queue_manager.acknowledge(task.id)
            return
        
        # Update state to RUNNING
        self._update_state(
            job_id=job_id,
            segment_id=segment_id,
            status=Status.RUNNING.value,
        )
        
        # Log start
        job_log = JobEventLog(self.state_dir, job_id)
        job_log.log_segment_started(segment_id or task.id, stage, self.worker_id)
        
        start_time = time.time()
        
        try:
            # Do the actual work
            result = self.process_task(task)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            if result.success:
                # Update state to COMPLETED
                self._update_state(
                    job_id=job_id,
                    segment_id=segment_id,
                    status=Status.COMPLETED.value,
                    output_paths=result.output_paths,
                    duration_seconds=duration_ms / 1000,
                )
                
                # Save completion checkpoint
                self._save_checkpoint(task, {"completed": True, "outputs": result.output_paths})
                
                # Log completion
                job_log.log_segment_completed(segment_id or task.id, stage, duration_ms)
                
                # Acknowledge task
                self.queue_manager.acknowledge(task.id)
                
                print(f"[{self.worker_id}] Task {task.id} completed in {duration_ms}ms")
                
            else:
                raise Exception(result.error or "Unknown error")
                
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            error_tb = traceback.format_exc()
            
            # Update state to FAILED
            self._update_state(
                job_id=job_id,
                segment_id=segment_id,
                status=Status.FAILED.value,
                error=error_msg,
            )
            
            # Log failure
            job_log.log_segment_failed(
                segment_id or task.id,
                stage,
                error_msg,
                task.attempt,
            )
            
            print(f"[{self.worker_id}] Task {task.id} failed: {error_msg}")
            
            # Reject task (will retry or move to DLQ)
            self.queue_manager.reject(task.id, error_msg, error_tb)
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def _update_state(
        self,
        job_id: str,
        segment_id: Optional[str],
        status: str,
        output_paths: Optional[list] = None,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Update state in FSM and SQLite"""
        stage = self.config.worker_type
        
        if segment_id:
            # Update segment state in FSM
            self.fsm.update_segment_state(
                job_id=job_id,
                stage=stage,
                segment_id=segment_id,
                new_status=status,
                output_paths=output_paths,
                error=error,
                duration_seconds=duration_seconds,
                worker_id=self.worker_id,
            )
            
            # Update SQLite
            self.database.update_segment_status(segment_id, status, error)
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def _get_checkpoint_path(self, task: Task) -> Path:
        """Get path for task checkpoint file"""
        return self.temp_dir / "checkpoints" / f"{task.id}.json"
    
    def _load_checkpoint(self, task: Task) -> Optional[dict]:
        """Load checkpoint for a task"""
        path = self._get_checkpoint_path(task)
        if path.exists():
            import json
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_checkpoint(self, task: Task, data: dict) -> None:
        """Save checkpoint for a task (atomic write)"""
        import json
        path = self._get_checkpoint_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(temp_path, path)
    
    def save_progress(self, task: Task, progress: dict) -> None:
        """
        Save intermediate progress during task processing.
        Call this periodically for long-running tasks.
        """
        checkpoint = self._load_checkpoint(task) or {}
        checkpoint["progress"] = progress
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_checkpoint(task, checkpoint)
    
    def load_progress(self, task: Task) -> Optional[dict]:
        """Load intermediate progress for resuming"""
        checkpoint = self._load_checkpoint(task)
        if checkpoint:
            return checkpoint.get("progress")
        return None
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[{self.worker_id}] Received shutdown signal, finishing current task...")
        self.running = False
    
    def get_temp_path(self, task: Task, suffix: str = "") -> Path:
        """Get a temp file path for task processing"""
        path = self.temp_dir / task.job_id / self.config.worker_type
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{task.id}{suffix}"


class WorkerPool:
    """
    Manages a pool of workers for a specific stage.
    
    Spawns multiple worker processes for parallel processing.
    """
    
    def __init__(
        self,
        worker_class: type,
        worker_config: WorkerConfig,
        num_workers: int,
        global_config: dict,
    ):
        self.worker_class = worker_class
        self.worker_config = worker_config
        self.num_workers = num_workers
        self.global_config = global_config
        
        self.workers: list = []
    
    def start(self):
        """Start all workers in the pool"""
        import multiprocessing
        
        for i in range(self.num_workers):
            config = WorkerConfig(
                worker_type=self.worker_config.worker_type,
                queue=self.worker_config.queue,
                worker_id=f"{self.worker_config.worker_type}_worker_{i}",
                gpu_device=self.worker_config.gpu_device,
            )
            
            # Each worker runs in its own process
            p = multiprocessing.Process(
                target=self._run_worker,
                args=(config,),
            )
            p.start()
            self.workers.append(p)
    
    def _run_worker(self, config: WorkerConfig):
        """Run a single worker (in subprocess)"""
        from tidaldub.state import AtomicFileState
        from tidaldub.state.database import StateDatabase
        from tidaldub.queues import QueueManager
        
        state_dir = self.global_config.get("paths", {}).get("state_dir", "./state")
        
        fsm = AtomicFileState(state_dir)
        database = StateDatabase(Path(state_dir) / "tidaldub.db")
        queue_manager = QueueManager(self.global_config)
        
        worker = self.worker_class(
            config=config,
            fsm=fsm,
            database=database,
            queue_manager=queue_manager,
            global_config=self.global_config,
        )
        
        worker.run()
    
    def stop(self):
        """Stop all workers"""
        for p in self.workers:
            p.terminate()
            p.join(timeout=5)
    
    def is_alive(self) -> bool:
        """Check if any worker is still running"""
        return any(p.is_alive() for p in self.workers)
