"""
TidalDub CLI
============

Command-line interface for the dubbing pipeline.

Commands:
    tidaldub submit <video> [options]   - Submit a video for dubbing
    tidaldub status <job_id>            - Check job status
    tidaldub resume <job_id>            - Resume a paused/crashed job
    tidaldub list                       - List all jobs
    tidaldub dlq [list|retry|skip]      - Manage dead letter queue
    tidaldub workers [start|stop]       - Manage workers
"""

import sys
from pathlib import Path
import uuid

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


console = Console()


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@click.group()
@click.version_option(version="0.1.0", prog_name="TidalDub")
def main():
    """TidalDub - Local Video Dubbing Pipeline"""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--audio-langs", "-a",
    default="es,fr,de",
    help="Comma-separated list of audio dubbing languages (default: es,fr,de)"
)
@click.option(
    "--subtitle-langs", "-s",
    default=None,
    help="Comma-separated list of subtitle languages (default: same as audio)"
)
@click.option(
    "--quality", "-q",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced",
    help="Quality preset (default: balanced)"
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume if a job for this video already exists"
)
def submit(video_path: str, audio_langs: str, subtitle_langs: str, quality: str, resume: bool):
    """Submit a video for dubbing."""
    config = load_config()
    
    video_path = Path(video_path).resolve()
    
    # Parse languages
    audio_languages = [lang.strip() for lang in audio_langs.split(",")]
    subtitle_languages = [lang.strip() for lang in (subtitle_langs or audio_langs).split(",")]
    
    console.print(f"\n[bold blue]TidalDub[/bold blue] - Video Dubbing Pipeline\n")
    console.print(f"[dim]Video:[/dim] {video_path}")
    console.print(f"[dim]Audio languages:[/dim] {', '.join(audio_languages)}")
    console.print(f"[dim]Subtitle languages:[/dim] {', '.join(subtitle_languages)}")
    console.print(f"[dim]Quality:[/dim] {quality}")
    console.print()
    
    # Initialize state management
    from tidaldub.state import AtomicFileState, JobState
    from tidaldub.state.database import StateDatabase
    from tidaldub.state.events import JobEventLog
    from tidaldub.queues import QueueManager, QueueName
    
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    queue_manager = QueueManager(config)
    
    # Generate job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    # Create job state
    job_state = JobState(
        job_id=job_id,
        video_path=str(video_path),
        audio_languages=audio_languages,
        subtitle_languages=subtitle_languages,
        quality_preset=quality,
    )
    
    # Create job in FSM
    fsm.create_job(job_state)
    
    # Create in database
    database.create_job(
        job_id=job_id,
        video_path=str(video_path),
        audio_languages=audio_languages,
        subtitle_languages=subtitle_languages,
        quality_preset=quality,
    )
    
    # Log job creation
    job_log = JobEventLog(state_dir, job_id)
    job_log.log_job_created(str(video_path), audio_languages, subtitle_languages)
    
    # Queue initial task (audio extraction / intake)
    queue_manager.enqueue(
        queue=QueueName.INTAKE,
        job_id=job_id,
        task_type="intake",
        payload={
            "video_path": str(video_path),
        }
    )
    
    console.print(f"[green]✓ Job submitted successfully![/green]")
    console.print(f"[dim]Job ID:[/dim] [bold]{job_id}[/bold]")
    console.print()
    console.print(f"[dim]Check status with:[/dim] tidaldub status {job_id}")
    console.print(f"[dim]Start workers with:[/dim] tidaldub workers start")


@main.command()
@click.argument("job_id")
def status(job_id: str):
    """Check the status of a job."""
    config = load_config()
    
    from tidaldub.state import AtomicFileState
    from tidaldub.state.database import StateDatabase
    
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    fsm = AtomicFileState(state_dir)
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    
    # Get job state
    job_state = fsm.get_job_state(job_id)
    
    if job_state is None:
        console.print(f"[red]Job not found: {job_id}[/red]")
        return
    
    # Display status
    console.print(f"\n[bold blue]Job Status[/bold blue]\n")
    
    # Create status table
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="dim")
    table.add_column("Value")
    
    table.add_row("Job ID", job_id)
    table.add_row("Video", job_state.video_path)
    table.add_row("Status", _format_status(job_state.status))
    table.add_row("Progress", f"{job_state.progress_percent:.1f}%")
    table.add_row("Current Stage", job_state.current_stage or "N/A")
    table.add_row("Audio Languages", ", ".join(job_state.audio_languages))
    table.add_row("Subtitle Languages", ", ".join(job_state.subtitle_languages))
    table.add_row("Created", job_state.created_at)
    
    if job_state.last_error:
        table.add_row("Last Error", f"[red]{job_state.last_error}[/red]")
    
    console.print(table)
    console.print()
    
    # Show stage progress
    console.print("[bold]Stage Progress[/bold]")
    
    stages = [
        ("01_intake", "Audio Extraction"),
        ("02_separation", "Source Separation"),
        ("03_transcription", "Transcription"),
        ("04_diarization", "Speaker Diarization"),
        ("05_translation", "Translation"),
        ("06_tts", "Voice Synthesis"),
        ("07_mixing", "Audio Mixing"),
        ("08_muxing", "Video Muxing"),
    ]
    
    for stage_id, stage_name in stages:
        stage_state = fsm.get_stage_state(job_id, stage_id)
        status = stage_state.get("status", "PENDING") if stage_state else "PENDING"
        icon = _status_icon(status)
        console.print(f"  {icon} {stage_name}: {status}")


@main.command("list")
@click.option("--status", "-s", default=None, help="Filter by status")
def list_jobs(status: str):
    """List all jobs."""
    config = load_config()
    
    from tidaldub.state.database import StateDatabase
    
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    
    jobs = database.list_jobs(status=status)
    
    if not jobs:
        console.print("[dim]No jobs found[/dim]")
        return
    
    table = Table(title="Jobs")
    table.add_column("Job ID")
    table.add_column("Video")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Created")
    
    for job in jobs:
        video_name = Path(job.video_path).name if job.video_path else "N/A"
        table.add_row(
            job.id,
            video_name[:30],
            _format_status(job.status),
            f"{job.progress_percent:.1f}%",
            str(job.created_at)[:16] if job.created_at else "N/A",
        )
    
    console.print(table)


@main.command()
@click.argument("job_id")
def resume(job_id: str):
    """Resume a paused or crashed job."""
    config = load_config()
    
    from tidaldub.recovery import RecoveryManager, create_recovery_manager
    
    recovery = create_recovery_manager(config)
    result = recovery.recover()
    
    console.print(f"[green]Recovery complete![/green]")
    console.print(f"[dim]Jobs recovered:[/dim] {result.jobs_recovered}")
    console.print(f"[dim]Segments requeued:[/dim] {result.segments_requeued}")


@main.group()
def dlq():
    """Manage dead letter queue."""
    pass


@dlq.command("list")
def dlq_list():
    """List items in the dead letter queue."""
    config = load_config()
    
    from tidaldub.state.database import StateDatabase
    
    state_dir = config.get("paths", {}).get("state_dir", "./state")
    database = StateDatabase(Path(state_dir) / "tidaldub.db")
    
    items = database.get_dlq_items()
    
    if not items:
        console.print("[green]Dead letter queue is empty![/green]")
        return
    
    table = Table(title="Dead Letter Queue")
    table.add_column("ID")
    table.add_column("Job")
    table.add_column("Task Type")
    table.add_column("Error")
    table.add_column("Retries")
    
    for item in items:
        table.add_row(
            item.id[:12],
            item.job_id[:12],
            item.task_type,
            (item.error_message or "")[:40],
            str(item.retry_count),
        )
    
    console.print(table)


@dlq.command("retry")
@click.argument("item_id")
def dlq_retry(item_id: str):
    """Retry a dead letter queue item."""
    config = load_config()
    
    from tidaldub.queues import QueueManager
    
    queue_manager = QueueManager(config)
    success = queue_manager.retry_dlq_item(item_id)
    
    if success:
        console.print(f"[green]Item requeued for retry[/green]")
    else:
        console.print(f"[red]Item not found: {item_id}[/red]")


@main.group()
def workers():
    """Manage worker processes."""
    pass


@workers.command("start")
@click.option("--worker", "-w", default="all", help="Worker type to start (or 'all')")
def workers_start(worker: str):
    """Start worker processes."""
    console.print("[yellow]Worker management not yet implemented.[/yellow]")
    console.print("[dim]Run workers individually with:[/dim]")
    console.print("  python workers/separation/worker.py")
    console.print("  python workers/transcription/worker.py")
    console.print("  python workers/diarization/worker.py")
    console.print("  python workers/translation/worker.py")
    console.print("  python workers/tts/worker.py")
    console.print("  python workers/mixing/worker.py")


def _format_status(status: str) -> str:
    """Format status with color"""
    colors = {
        "PENDING": "white",
        "QUEUED": "cyan",
        "RUNNING": "yellow",
        "COMPLETED": "green",
        "FAILED": "red",
        "DEAD_LETTER": "red",
    }
    color = colors.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _status_icon(status: str) -> str:
    """Get status icon"""
    icons = {
        "PENDING": "○",
        "QUEUED": "◐",
        "RUNNING": "◑",
        "COMPLETED": "●",
        "FAILED": "✗",
    }
    return icons.get(status, "○")


if __name__ == "__main__":
    main()
