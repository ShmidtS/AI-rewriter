"""
Data contracts for the Application Service Layer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RewriteStatus(str, Enum):
    """Status of a rewrite job."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ProgressInfo:
    """Progress information for a rewrite job."""

    current: int = 0
    total: int = 0
    status: RewriteStatus = RewriteStatus.IDLE
    filename: str = ""
    output_name: str = ""
    error_message: str | None = None

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total <= 0:
            return 0.0
        return round((self.current / self.total) * 100, 1)

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == RewriteStatus.RUNNING

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "status": self.status.value,
            "filename": self.filename,
            "output_name": self.output_name,
            "error_message": self.error_message,
            "is_running": self.is_running,
        }


@dataclass
class RewriteParams:
    """Parameters for a rewrite job."""

    input_file: str
    output_file: str
    language: str = "Русский"
    style: str = ""
    goal: str = ""
    model: str = ""
    resume: bool = True
    save_interval: int = 1
    prompt_preset: str = "literary"
    base_url: str | None = None
    token: str | None = None
    parallel: bool = False
    max_workers: int | None = 10
    output_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for legacy rewrite_process compatibility."""
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "language": self.language,
            "style": self.style,
            "goal": self.goal,
            "rewriter_model": self.model,
            "resume": self.resume,
            "save_interval": self.save_interval,
            "prompt_preset": self.prompt_preset,
            "base_url": self.base_url,
            "token": self.token,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "output_language": self.output_language,
        }

