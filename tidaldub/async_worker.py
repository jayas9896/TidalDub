"""
Async Worker Framework
======================

High-performance async worker implementation for maximum throughput.

Features:
- Non-blocking queue processing with Redis pub/sub or SQLite WAL
- Concurrent task handling for CPU-bound workers
- torch.compile integration for 2-3x faster inference
- Hardware-optimized settings
"""

import asyncio
import os
import sys
import time
import signal
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass
import multiprocessing as mp

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidaldub.state import AtomicFileState, Status
from tidaldub.state.database import StateDatabase
from tidaldub.state.events import JobEventLog
from tidaldub.queues import QueueManager, Task, QueueName


@dataclass
class AsyncWorkerConfig:
    """Configuration for async workers"""
    worker_type: str
    queue: QueueName
    
    # Concurrency settings
    max_concurrent_tasks: int = 1  # GPU workers: 1, CPU workers: 4+
    use_process_pool: bool = False  # True for CPU-bound, False for GPU
    
    # GPU settings
    gpu_device: str = "cuda:0"
    use_torch_compile: bool = True  # Enable torch.compile for faster inference
    use_flash_attention: bool = True  # Enable Flash Attention 2
    
    # Queue settings
    poll_interval_ms: int = 100  # Fast polling when no Redis pub/sub
    batch_size: int = 1  # Process multiple tasks at once (for batching)


class AsyncQueueListener:
    """
    Async queue listener with instant task notification.
    
    Uses Redis pub/sub when available, falls back to fast polling.
    """
    
    def __init__(self, queue_manager: QueueManager, queue: QueueName):
        self.queue_manager = queue_manager
        self.queue = queue
        self._running = False
        self._has_redis_pubsub = False
        
        # Check if Redis pub/sub is available
        if queue_manager.use_redis and hasattr(queue_manager.backend, 'redis'):
            self._has_redis_pubsub = True
    
    async def subscribe(self):
        """
        Async generator that yields tasks as they arrive.
        Uses pub/sub for instant notification when available.
        """
        self._running = True
        
        if self._has_redis_pubsub:
            async for task in self._redis_subscribe():
                yield task
        else:
            async for task in self._poll_subscribe():
                yield task
    
    async def _redis_subscribe(self):
        """Subscribe via Redis pub/sub for instant notifications"""
        import redis.asyncio as aioredis
        
        redis_url = self.queue_manager.config.get("queues", {}).get(
            "redis_url", "redis://localhost:6379/0"
        )
        
        redis = await aioredis.from_url(redis_url)
        pubsub = redis.pubsub()
        
        # Subscribe to notification channel
        channel = f"{self.queue.value}:notify"
        await pubsub.subscribe(channel)
        
        while self._running:
            try:
                # Wait for notification (with timeout for graceful shutdown)
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                    timeout=5.0
                )
                
                if message:
                    # Notification received, fetch task from queue
                    task = self.queue_manager.dequeue(
                        self.queue,
                        worker_id=f"async_{os.getpid()}",
                        timeout=0,  # Non-blocking
                    )
                    if task:
                        yield task
                        
            except asyncio.TimeoutError:
                # Check for any pending tasks
                task = self.queue_manager.dequeue(
                    self.queue,
                    worker_id=f"async_{os.getpid()}",
                    timeout=0,
                )
                if task:
                    yield task
            except Exception as e:
                print(f"Redis subscribe error: {e}")
                await asyncio.sleep(1)
        
        await pubsub.unsubscribe(channel)
        await redis.close()
    
    async def _poll_subscribe(self):
        """Fallback: Fast polling for SQLite backend"""
        while self._running:
            task = self.queue_manager.dequeue(
                self.queue,
                worker_id=f"async_{os.getpid()}",
                timeout=0,  # Non-blocking
            )
            
            if task:
                yield task
            else:
                # Short sleep to avoid busy-waiting
                await asyncio.sleep(0.1)  # 100ms polling
    
    def stop(self):
        self._running = False


class AsyncWorker(ABC):
    """
    Async worker base class for high-throughput processing.
    
    Features:
    - Async task processing with concurrent execution
    - torch.compile integration for faster inference
    - Process pool for CPU-bound parallel work
    """
    
    def __init__(
        self,
        config: AsyncWorkerConfig,
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
        
        self.worker_id = f"{config.worker_type}_{os.getpid()}"
        self.running = False
        
        # Paths
        self.state_dir = Path(global_config.get("paths", {}).get("state_dir", "./state"))
        self.data_dir = Path(global_config.get("paths", {}).get("data_dir", "./data"))
        
        # Executor for parallel CPU work
        self._executor: Optional[ProcessPoolExecutor] = None
        
        # Queue listener
        self._listener: Optional[AsyncQueueListener] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the worker (load models, etc.)"""
        pass
    
    @abstractmethod
    async def process_task(self, task: Task) -> dict:
        """Process a single task, return result dict"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources"""
        pass
    
    async def run(self) -> None:
        """Main async worker loop"""
        print(f"[{self.worker_id}] Starting async worker...")
        
        try:
            await self.initialize()
            self.running = True
            
            # Create executor for parallel work if configured
            if self.config.use_process_pool and self.config.max_concurrent_tasks > 1:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.config.max_concurrent_tasks
                )
            
            # Create queue listener
            self._listener = AsyncQueueListener(self.queue_manager, self.config.queue)
            
            print(f"[{self.worker_id}] Worker ready, listening on {self.config.queue.value}")
            
            if self.config.max_concurrent_tasks > 1:
                await self._run_concurrent()
            else:
                await self._run_sequential()
                
        except Exception as e:
            print(f"[{self.worker_id}] Fatal error: {e}")
            traceback.print_exc()
        finally:
            await self.cleanup()
            if self._executor:
                self._executor.shutdown(wait=True)
            print(f"[{self.worker_id}] Worker stopped")
    
    async def _run_sequential(self) -> None:
        """Run tasks one at a time (for GPU workers)"""
        async for task in self._listener.subscribe():
            if not self.running:
                break
            
            try:
                await self._process_with_tracking(task)
            except Exception as e:
                print(f"[{self.worker_id}] Task error: {e}")
    
    async def _run_concurrent(self) -> None:
        """Run multiple tasks concurrently (for CPU workers)"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        tasks = set()
        
        async def process_with_semaphore(task: Task):
            async with semaphore:
                await self._process_with_tracking(task)
        
        async for task in self._listener.subscribe():
            if not self.running:
                break
            
            # Create task for concurrent processing
            coro = process_with_semaphore(task)
            t = asyncio.create_task(coro)
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        
        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_with_tracking(self, task: Task) -> None:
        """Process task with state tracking and error handling"""
        job_id = task.job_id
        segment_id = task.payload.get("segment_id")
        stage = self.config.worker_type
        
        # Update state to RUNNING
        self._update_state(job_id, segment_id, Status.RUNNING.value)
        
        # Log start
        job_log = JobEventLog(self.state_dir, job_id)
        job_log.log_segment_started(segment_id or task.id, stage, self.worker_id)
        
        start_time = time.time()
        
        try:
            result = await self.process_task(task)
            duration_ms = int((time.time() - start_time) * 1000)
            
            if result.get("success", False):
                self._update_state(
                    job_id, segment_id, Status.COMPLETED.value,
                    output_paths=result.get("output_paths"),
                    duration_seconds=duration_ms / 1000,
                )
                job_log.log_segment_completed(segment_id or task.id, stage, duration_ms)
                self.queue_manager.acknowledge(task.id)
                print(f"[{self.worker_id}] ✓ {task.id} completed in {duration_ms}ms")
            else:
                raise Exception(result.get("error", "Unknown error"))
                
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            self._update_state(job_id, segment_id, Status.FAILED.value, error=error_msg)
            job_log.log_segment_failed(segment_id or task.id, stage, error_msg, task.attempt)
            
            print(f"[{self.worker_id}] ✗ {task.id} failed: {error_msg}")
            self.queue_manager.reject(task.id, error_msg, traceback.format_exc())
    
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
            self.database.update_segment_status(segment_id, status, error)
    
    def stop(self):
        """Stop the worker gracefully"""
        self.running = False
        if self._listener:
            self._listener.stop()


def setup_torch_compile(model, mode: str = "reduce-overhead"):
    """
    Apply torch.compile for faster inference.
    
    Modes:
    - "default": Good balance
    - "reduce-overhead": Best for inference (2-3x speedup)
    - "max-autotune": Maximum performance (longer compile time)
    """
    import torch
    
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    try:
        compiled = torch.compile(model, mode=mode)
        print(f"torch.compile enabled with mode={mode}")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}")
        return model


def setup_flash_attention():
    """Enable Flash Attention 2 for memory-efficient attention"""
    import torch
    
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable Flash Attention via SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        print("Flash Attention 2 enabled")


def get_optimal_worker_count(worker_type: str, cpu_cores: int = None) -> int:
    """
    Get optimal worker count based on hardware.
    
    For your Intel Core Ultra 9 275HX (24 cores):
    - GPU workers: 1 (VRAM limited)
    - CPU workers: 4-8 (leave cores for system)
    """
    if cpu_cores is None:
        cpu_cores = os.cpu_count() or 4
    
    if worker_type in ["separation", "transcription", "diarization", "translation", "tts"]:
        # GPU-bound workers - one at a time due to VRAM
        return 1
    elif worker_type == "mixing":
        # CPU-bound - can parallelize
        return min(8, max(4, cpu_cores // 3))  # Use 1/3 of cores
    else:
        return 1
