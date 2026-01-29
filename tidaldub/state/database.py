"""
SQLite State Database
=====================

Fast queryable cache for state data. Can be fully rebuilt from the
file-based FSM if corrupted.

This provides:
- Indexed queries for progress tracking
- Analytics and reporting
- Fast lookups without scanning files
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import json

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    event,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
)


Base = declarative_base()


class Job(Base):
    """Main job tracking table"""
    __tablename__ = "jobs"
    
    id = Column(String(64), primary_key=True)
    video_path = Column(Text, nullable=False)
    status = Column(String(32), default="PENDING", index=True)
    
    # Configuration (stored as JSON strings)
    audio_languages = Column(Text)  # JSON array
    subtitle_languages = Column(Text)  # JSON array
    quality_preset = Column(String(32), default="balanced")
    
    # Progress
    current_stage = Column(String(64))
    progress_percent = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))
    
    # Error tracking
    last_error = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    segments = relationship("Segment", back_populates="job", cascade="all, delete-orphan")
    language_progress = relationship("LanguageProgress", back_populates="job", cascade="all, delete-orphan")
    dlq_items = relationship("DeadLetterItem", back_populates="job", cascade="all, delete-orphan")
    
    def get_audio_languages(self) -> List[str]:
        return json.loads(self.audio_languages) if self.audio_languages else []
    
    def set_audio_languages(self, langs: List[str]):
        self.audio_languages = json.dumps(langs)
    
    def get_subtitle_languages(self) -> List[str]:
        return json.loads(self.subtitle_languages) if self.subtitle_languages else []
    
    def set_subtitle_languages(self, langs: List[str]):
        self.subtitle_languages = json.dumps(langs)


class Segment(Base):
    """Granular segment tracking for chunk-level recovery"""
    __tablename__ = "segments"
    
    id = Column(String(128), primary_key=True)
    job_id = Column(String(64), ForeignKey("jobs.id"), nullable=False, index=True)
    
    # Segment info
    segment_index = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)
    speaker_id = Column(String(64))
    language = Column(String(8))  # For per-language stages
    
    # Processing stage
    stage = Column(String(64), index=True)
    status = Column(String(32), default="PENDING", index=True)
    
    # Retry tracking
    retry_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Checkpoint
    checkpoint_path = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationship
    job = relationship("Job", back_populates="segments")
    
    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_segment_job_stage", "job_id", "stage"),
        Index("ix_segment_job_stage_status", "job_id", "stage", "status"),
    )


class LanguageProgress(Base):
    """Per-language progress tracking"""
    __tablename__ = "language_progress"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("jobs.id"), nullable=False)
    language = Column(String(8), nullable=False)
    stage = Column(String(64), nullable=False)  # translation, tts, mixing
    
    segments_completed = Column(Integer, default=0)
    segments_total = Column(Integer)
    status = Column(String(32), default="PENDING")
    
    job = relationship("Job", back_populates="language_progress")
    
    __table_args__ = (
        Index("ix_lang_progress_job_lang", "job_id", "language"),
    )


class DeadLetterItem(Base):
    """Dead letter queue for failed tasks"""
    __tablename__ = "dead_letter_queue"
    
    id = Column(String(128), primary_key=True)
    job_id = Column(String(64), ForeignKey("jobs.id"), nullable=False, index=True)
    segment_id = Column(String(128))
    
    task_type = Column(String(64), nullable=False)
    payload = Column(Text)  # JSON: full task details
    
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    retry_count = Column(Integer, default=0)
    
    # Resolution
    resolution_status = Column(String(32), default="unresolved", index=True)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Timestamps
    failed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    job = relationship("Job", back_populates="dlq_items")


class RecoveryLog(Base):
    """System recovery event log"""
    __tablename__ = "recovery_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False)  # startup, shutdown, crash_detected, recovery_complete
    job_id = Column(String(64))  # Optional, if related to a specific job
    details = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class StateDatabase:
    """
    SQLite database for fast state queries.
    
    This is a CACHE that can be rebuilt from the file-based FSM.
    All writes should go to FSM first, then here.
    """
    
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite connection string
        db_url = f"sqlite:///{self.db_path}"
        
        self.engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Enable WAL mode for better concurrent access
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.Session()
    
    # =========================================================================
    # Job Operations
    # =========================================================================
    
    def create_job(
        self,
        job_id: str,
        video_path: str,
        audio_languages: List[str],
        subtitle_languages: List[str],
        quality_preset: str = "balanced",
    ) -> Job:
        """Create a new job in the database"""
        with self.get_session() as session:
            job = Job(
                id=job_id,
                video_path=video_path,
                audio_languages=json.dumps(audio_languages),
                subtitle_languages=json.dumps(subtitle_languages),
                quality_preset=quality_preset,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        with self.get_session() as session:
            return session.query(Job).filter(Job.id == job_id).first()
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        current_stage: Optional[str] = None,
        progress_percent: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update job status"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = status
                job.updated_at = datetime.now(timezone.utc)
                if current_stage:
                    job.current_stage = current_stage
                if progress_percent is not None:
                    job.progress_percent = progress_percent
                if error:
                    job.last_error = error
                    job.retry_count += 1
                session.commit()
    
    def list_jobs(self, status: Optional[str] = None) -> List[Job]:
        """List all jobs, optionally filtered by status"""
        with self.get_session() as session:
            query = session.query(Job)
            if status:
                query = query.filter(Job.status == status)
            return query.order_by(Job.created_at.desc()).all()
    
    def list_incomplete_jobs(self) -> List[Job]:
        """List jobs that need to be resumed"""
        with self.get_session() as session:
            return session.query(Job).filter(
                Job.status.in_(["PENDING", "RUNNING", "QUEUED", "RETRYING"])
            ).all()
    
    # =========================================================================
    # Segment Operations
    # =========================================================================
    
    def create_segment(
        self,
        segment_id: str,
        job_id: str,
        stage: str,
        segment_index: int,
        start_time: float,
        end_time: float,
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Segment:
        """Create a new segment"""
        with self.get_session() as session:
            segment = Segment(
                id=segment_id,
                job_id=job_id,
                stage=stage,
                segment_index=segment_index,
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                language=language,
            )
            session.add(segment)
            session.commit()
            return segment
    
    def update_segment_status(
        self,
        segment_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update segment status"""
        with self.get_session() as session:
            segment = session.query(Segment).filter(Segment.id == segment_id).first()
            if segment:
                segment.status = status
                segment.updated_at = datetime.now(timezone.utc)
                if error:
                    segment.last_error = error
                    segment.retry_count += 1
                session.commit()
    
    def get_incomplete_segments(
        self,
        job_id: str,
        stage: str,
    ) -> List[Segment]:
        """Get segments that are not completed for a stage"""
        with self.get_session() as session:
            return session.query(Segment).filter(
                Segment.job_id == job_id,
                Segment.stage == stage,
                Segment.status != "COMPLETED",
            ).order_by(Segment.segment_index).all()
    
    def get_stage_progress(
        self,
        job_id: str,
        stage: str,
    ) -> dict:
        """Get completion progress for a stage"""
        with self.get_session() as session:
            total = session.query(Segment).filter(
                Segment.job_id == job_id,
                Segment.stage == stage,
            ).count()
            
            completed = session.query(Segment).filter(
                Segment.job_id == job_id,
                Segment.stage == stage,
                Segment.status == "COMPLETED",
            ).count()
            
            return {
                "total": total,
                "completed": completed,
                "percent": (completed / total * 100) if total > 0 else 0,
            }
    
    # =========================================================================
    # Dead Letter Queue Operations
    # =========================================================================
    
    def add_to_dlq(
        self,
        item_id: str,
        job_id: str,
        task_type: str,
        error_message: str,
        error_traceback: Optional[str] = None,
        segment_id: Optional[str] = None,
        payload: Optional[dict] = None,
        retry_count: int = 0,
    ) -> DeadLetterItem:
        """Add an item to the dead letter queue"""
        with self.get_session() as session:
            item = DeadLetterItem(
                id=item_id,
                job_id=job_id,
                segment_id=segment_id,
                task_type=task_type,
                payload=json.dumps(payload) if payload else None,
                error_message=error_message,
                error_traceback=error_traceback,
                retry_count=retry_count,
            )
            session.add(item)
            session.commit()
            return item
    
    def get_dlq_items(
        self,
        job_id: Optional[str] = None,
        status: str = "unresolved",
    ) -> List[DeadLetterItem]:
        """Get dead letter queue items"""
        with self.get_session() as session:
            query = session.query(DeadLetterItem).filter(
                DeadLetterItem.resolution_status == status
            )
            if job_id:
                query = query.filter(DeadLetterItem.job_id == job_id)
            return query.order_by(DeadLetterItem.failed_at.desc()).all()
    
    def resolve_dlq_item(
        self,
        item_id: str,
        resolution: str,  # retried, skipped, manual
        notes: Optional[str] = None,
    ) -> None:
        """Mark a DLQ item as resolved"""
        with self.get_session() as session:
            item = session.query(DeadLetterItem).filter(
                DeadLetterItem.id == item_id
            ).first()
            if item:
                item.resolution_status = resolution
                item.resolved_at = datetime.now(timezone.utc)
                item.resolution_notes = notes
                session.commit()
    
    # =========================================================================
    # Recovery Operations
    # =========================================================================
    
    def log_recovery_event(
        self,
        event_type: str,
        job_id: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log a recovery event"""
        with self.get_session() as session:
            log = RecoveryLog(
                event_type=event_type,
                job_id=job_id,
                details=details,
            )
            session.add(log)
            session.commit()
    
    def rebuild_from_fsm(self, fsm: "AtomicFileState") -> None:
        """
        Rebuild the entire database from file-based FSM.
        Called when database is corrupted or missing.
        """
        # Drop all data
        with self.get_session() as session:
            session.query(Segment).delete()
            session.query(LanguageProgress).delete()
            session.query(DeadLetterItem).delete()
            session.query(Job).delete()
            session.commit()
        
        # Rebuild from FSM files
        for job_id in fsm.list_jobs():
            job_state = fsm.get_job_state(job_id)
            if job_state:
                self.create_job(
                    job_id=job_state.job_id,
                    video_path=job_state.video_path,
                    audio_languages=job_state.audio_languages,
                    subtitle_languages=job_state.subtitle_languages,
                    quality_preset=job_state.quality_preset,
                )
                self.update_job_status(
                    job_id=job_state.job_id,
                    status=job_state.status,
                    current_stage=job_state.current_stage,
                    progress_percent=job_state.progress_percent,
                )
        
        self.log_recovery_event("database_rebuilt", details="Rebuilt from FSM")


def create_database(config: dict) -> StateDatabase:
    """Create database from config"""
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    db_path = Path(state_dir) / "tidaldub.db"
    return StateDatabase(db_path)
