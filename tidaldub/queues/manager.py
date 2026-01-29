"""
Queue Manager
=============

Manages task queues for the dubbing pipeline.
Supports Redis for fast messaging with SQLite fallback.
Includes dead letter queue (DLQ) for failed tasks.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class QueueName(str, Enum):
    """Standard queue names for each pipeline stage"""
    INTAKE = "tidaldub:intake"
    SEPARATION = "tidaldub:separation"
    TRANSCRIPTION = "tidaldub:transcription"
    DIARIZATION = "tidaldub:diarization"
    TRANSLATION = "tidaldub:translation"
    TTS = "tidaldub:tts"
    MIXING = "tidaldub:mixing"
    MUXING = "tidaldub:muxing"
    
    # Special queues
    DEAD_LETTER = "tidaldub:dlq"
    PRIORITY = "tidaldub:priority"


@dataclass
class Task:
    """A task in the queue"""
    id: str
    queue: str
    job_id: str
    task_type: str
    payload: dict = field(default_factory=dict)
    
    # Retry tracking
    attempt: int = 1
    max_attempts: int = 3
    
    # Timing
    created_at: str = ""
    started_at: Optional[str] = None
    
    # Worker info
    worker_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> "Task":
        return cls(**json.loads(data))


@dataclass
class DLQItem:
    """An item in the dead letter queue"""
    id: str
    original_task: Task
    error_message: str
    error_traceback: Optional[str] = None
    failed_at: str = ""
    resolution_status: str = "unresolved"  # unresolved, retried, skipped, manual
    
    def __post_init__(self):
        if not self.failed_at:
            self.failed_at = datetime.now(timezone.utc).isoformat()


class RedisQueueBackend:
    """Redis-based queue implementation with pub/sub notifications"""
    
    def __init__(self, redis_url: str):
        if not HAS_REDIS:
            raise ImportError("redis package not installed")
        
        self.redis = redis.from_url(redis_url)
        self.redis_url = redis_url
        
    def push(self, queue: str, task: Task) -> None:
        """Push a task to the queue and notify listeners"""
        self.redis.rpush(queue, task.to_json())
        # Publish notification for instant pickup
        self.redis.publish(f"{queue}:notify", "new_task")
    
    def push_priority(self, queue: str, task: Task) -> None:
        """Push a task to the front of the queue (high priority)"""
        self.redis.lpush(queue, task.to_json())
        self.redis.publish(f"{queue}:notify", "new_task")
    
    def pop(self, queue: str, timeout: int = 0) -> Optional[Task]:
        """Pop a task from the queue (blocking)"""
        result = self.redis.blpop(queue, timeout=timeout)
        if result:
            _, data = result
            return Task.from_json(data.decode())
        return None
    
    def pop_nonblocking(self, queue: str) -> Optional[Task]:
        """Pop a task without blocking"""
        data = self.redis.lpop(queue)
        if data:
            return Task.from_json(data.decode())
        return None
    
    def length(self, queue: str) -> int:
        """Get queue length"""
        return self.redis.llen(queue)
    
    def peek(self, queue: str, count: int = 10) -> list[Task]:
        """Peek at tasks without removing them"""
        items = self.redis.lrange(queue, 0, count - 1)
        return [Task.from_json(item.decode()) for item in items]
    
    def clear(self, queue: str) -> None:
        """Clear all items from queue"""
        self.redis.delete(queue)


class SQLiteQueueBackend:
    """SQLite-based queue fallback (no Redis dependency)"""
    
    def __init__(self, db_path: Path):
        from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
        from sqlalchemy.orm import declarative_base, sessionmaker
        
        self.db_path = db_path
        db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url, echo=False)
        
        Base = declarative_base()
        
        class QueueItem(Base):
            __tablename__ = "queue_items"
            id = Column(String(64), primary_key=True)
            queue = Column(String(64), index=True)
            data = Column(Text)
            priority = Column(Integer, default=0)
            created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        
        self.QueueItem = QueueItem
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def push(self, queue: str, task: Task) -> None:
        with self.Session() as session:
            item = self.QueueItem(
                id=task.id,
                queue=queue,
                data=task.to_json(),
                priority=0,
            )
            session.add(item)
            session.commit()
    
    def push_priority(self, queue: str, task: Task) -> None:
        with self.Session() as session:
            item = self.QueueItem(
                id=task.id,
                queue=queue,
                data=task.to_json(),
                priority=1,  # Higher priority
            )
            session.add(item)
            session.commit()
    
    def pop(self, queue: str, timeout: int = 0) -> Optional[Task]:
        """Pop with optional polling (not true blocking)"""
        start_time = time.time()
        
        while True:
            task = self.pop_nonblocking(queue)
            if task:
                return task
            
            if timeout == 0:
                return None
            
            if time.time() - start_time >= timeout:
                return None
            
            time.sleep(0.1)  # Poll interval
    
    def pop_nonblocking(self, queue: str) -> Optional[Task]:
        with self.Session() as session:
            item = session.query(self.QueueItem).filter(
                self.QueueItem.queue == queue
            ).order_by(
                self.QueueItem.priority.desc(),
                self.QueueItem.created_at.asc(),
            ).first()
            
            if item:
                data = item.data
                session.delete(item)
                session.commit()
                return Task.from_json(data)
            
            return None
    
    def length(self, queue: str) -> int:
        with self.Session() as session:
            return session.query(self.QueueItem).filter(
                self.QueueItem.queue == queue
            ).count()
    
    def peek(self, queue: str, count: int = 10) -> list[Task]:
        with self.Session() as session:
            items = session.query(self.QueueItem).filter(
                self.QueueItem.queue == queue
            ).order_by(
                self.QueueItem.priority.desc(),
                self.QueueItem.created_at.asc(),
            ).limit(count).all()
            
            return [Task.from_json(item.data) for item in items]
    
    def clear(self, queue: str) -> None:
        with self.Session() as session:
            session.query(self.QueueItem).filter(
                self.QueueItem.queue == queue
            ).delete()
            session.commit()


class QueueManager:
    """
    Manages task queues for the dubbing pipeline.
    
    Features:
    - Redis backend for speed (with SQLite fallback)
    - Dead letter queue for failed tasks
    - Retry with exponential backoff
    - Task acknowledgment pattern
    """
    
    def __init__(self, config: dict):
        self.config = config
        queue_config = config.get("queues", {})
        
        self.use_redis = queue_config.get("use_redis", False) and HAS_REDIS
        self.max_retries = queue_config.get("max_retries", 3)
        self.retry_delay_base = queue_config.get("retry_delay_base_sec", 10)
        
        # Initialize backend
        if self.use_redis:
            redis_url = queue_config.get("redis_url", "redis://localhost:6379/0")
            self.backend = RedisQueueBackend(redis_url)
        else:
            state_dir = config.get("paths", {}).get("state_dir", "./state")
            db_path = Path(state_dir) / "queues.db"
            self.backend = SQLiteQueueBackend(db_path)
        
        # In-progress tasks (for acknowledgment)
        self._in_progress: dict[str, Task] = {}
    
    # =========================================================================
    # Basic Queue Operations
    # =========================================================================
    
    def enqueue(
        self,
        queue: QueueName | str,
        job_id: str,
        task_type: str,
        payload: dict,
        priority: bool = False,
    ) -> Task:
        """Add a task to the queue"""
        queue_name = queue.value if isinstance(queue, QueueName) else queue
        
        task = Task(
            id=str(uuid.uuid4()),
            queue=queue_name,
            job_id=job_id,
            task_type=task_type,
            payload=payload,
            max_attempts=self.max_retries,
        )
        
        if priority:
            self.backend.push_priority(queue_name, task)
        else:
            self.backend.push(queue_name, task)
        
        return task
    
    def dequeue(
        self,
        queue: QueueName | str,
        worker_id: str,
        timeout: int = 30,
    ) -> Optional[Task]:
        """
        Get a task from the queue.
        
        The task is moved to in-progress state until acknowledged.
        """
        queue_name = queue.value if isinstance(queue, QueueName) else queue
        
        task = self.backend.pop(queue_name, timeout=timeout)
        
        if task:
            task.worker_id = worker_id
            task.started_at = datetime.now(timezone.utc).isoformat()
            self._in_progress[task.id] = task
        
        return task
    
    def acknowledge(self, task_id: str) -> None:
        """Mark a task as successfully completed"""
        self._in_progress.pop(task_id, None)
    
    def reject(
        self,
        task_id: str,
        error: str,
        traceback: Optional[str] = None,
        requeue: bool = True,
    ) -> None:
        """
        Reject a task (failed processing).
        
        If requeue=True and attempts remaining, requeue with backoff.
        Otherwise, move to dead letter queue.
        """
        task = self._in_progress.pop(task_id, None)
        
        if task is None:
            return
        
        if requeue and task.attempt < task.max_attempts:
            # Retry with exponential backoff
            task.attempt += 1
            
            # In a real implementation, we'd use a delayed queue
            # For now, just requeue immediately
            self.backend.push(task.queue, task)
        else:
            # Move to dead letter queue
            self._send_to_dlq(task, error, traceback)
    
    def _send_to_dlq(
        self,
        task: Task,
        error: str,
        traceback: Optional[str] = None,
    ) -> None:
        """Send a task to the dead letter queue"""
        dlq_item = DLQItem(
            id=f"dlq_{task.id}",
            original_task=task,
            error_message=error,
            error_traceback=traceback,
        )
        
        # Store in DLQ
        dlq_task = Task(
            id=dlq_item.id,
            queue=QueueName.DEAD_LETTER.value,
            job_id=task.job_id,
            task_type="dlq_item",
            payload={
                "original_task": asdict(task),
                "error_message": error,
                "error_traceback": traceback,
            }
        )
        
        self.backend.push(QueueName.DEAD_LETTER.value, dlq_task)
    
    # =========================================================================
    # Queue Status
    # =========================================================================
    
    def get_queue_lengths(self) -> dict[str, int]:
        """Get the length of all queues"""
        return {
            queue.name: self.backend.length(queue.value)
            for queue in QueueName
        }
    
    def get_in_progress_count(self) -> int:
        """Get count of tasks currently being processed"""
        return len(self._in_progress)
    
    def peek_queue(self, queue: QueueName | str, count: int = 10) -> list[Task]:
        """Peek at tasks in a queue without removing them"""
        queue_name = queue.value if isinstance(queue, QueueName) else queue
        return self.backend.peek(queue_name, count)
    
    # =========================================================================
    # DLQ Operations
    # =========================================================================
    
    def get_dlq_items(self, count: int = 100) -> list[Task]:
        """Get items from the dead letter queue"""
        return self.backend.peek(QueueName.DEAD_LETTER.value, count)
    
    def retry_dlq_item(self, dlq_task_id: str) -> bool:
        """Retry a dead letter queue item"""
        # This is a simplified implementation
        # In production, you'd want to properly pop and requeue
        dlq_items = self.backend.peek(QueueName.DEAD_LETTER.value, 1000)
        
        for item in dlq_items:
            if item.id == dlq_task_id:
                # Reconstruct original task
                original = item.payload.get("original_task", {})
                if original:
                    task = Task(**original)
                    task.attempt = 1  # Reset attempts
                    self.backend.push(task.queue, task)
                    return True
        
        return False
    
    def clear_all_queues(self) -> None:
        """Clear all queues (use with caution!)"""
        for queue in QueueName:
            self.backend.clear(queue.value)
        self._in_progress.clear()


def create_queue_manager(config: dict) -> QueueManager:
    """Create queue manager from config"""
    return QueueManager(config)
