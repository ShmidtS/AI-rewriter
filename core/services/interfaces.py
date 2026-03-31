"""
Abstract interfaces for the Application Service Layer.

Defines contracts for:
- IRewriteService: High-level rewrite orchestration
- IModelProvider: Model discovery and validation
- IProgressReporter: Progress and status reporting
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
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


class IProgressReporter(ABC):
    """Interface for progress reporting during rewrite."""

    @abstractmethod
    def report_progress(self, current: int, total: int) -> None:
        """Report progress update."""
        pass

    @abstractmethod
    def report_log(self, message: str) -> None:
        """Report log message."""
        pass

    @abstractmethod
    def report_status(self, status: RewriteStatus, message: str | None = None) -> None:
        """Report status change."""
        pass


class IModelProvider(ABC):
    """Interface for model discovery and validation."""

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        pass

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """
        Validate connection to the API.

        Returns:
            Tuple of (success, message) where message contains error details if failed.
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model name."""
        pass


class IRewriteService(ABC):
    """Interface for high-level rewrite orchestration."""

    @abstractmethod
    def start_rewrite(
        self,
        params: RewriteParams,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> bool:
        """
        Start a rewrite job.

        Args:
            params: Rewrite parameters
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for log messages

        Returns:
            True if job started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_rewrite(self) -> None:
        """Stop the current rewrite job."""
        pass

    @abstractmethod
    def get_status(self) -> ProgressInfo:
        """Get current job status and progress."""
        pass

    @abstractmethod
    def get_progress(self) -> tuple[int, int]:
        """Get current progress as (current, total) tuple."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if a job is currently running."""
        pass

    @abstractmethod
    def wait_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for the current job to complete.

        Args:
            timeout: Maximum time to wait in seconds, None for indefinite

        Returns:
            True if job completed, False if timeout reached
        """
        pass
