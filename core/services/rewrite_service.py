"""
RewriteService - High-level facade for rewrite orchestration.

Provides clean API for GUI/Web interfaces while maintaining
backward compatibility with core.rewriter.rewrite_process().
"""

import logging
import threading
from collections.abc import Callable

from core.rewriter import rewrite_process
from core.services.interfaces import (
    IRewriteService,
    ProgressInfo,
    RewriteParams,
    RewriteStatus,
)
from core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class RewriteService(IRewriteService):
    """
    High-level service for managing rewrite operations.

    Acts as a facade over core.rewriter.rewrite_process(), providing:
    - Clean API for GUI/Web
    - Integration with typed Settings
    - Thread-safe job management
    - Progress and status tracking
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the rewrite service.

        Args:
            settings: Optional Settings instance. Uses global settings if not provided.
        """
        self._settings = settings or get_settings()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress: ProgressInfo = ProgressInfo()
        self._lock = threading.Lock()
        self._progress_callback: Callable[[int, int], None] | None = None
        self._log_callback: Callable[[str], None] | None = None

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
        with self._lock:
            if self._progress.is_running:
                logger.warning("Attempted to start rewrite while job is running")
                return False

            # Reset state
            self._stop_event.clear()
            self._progress_callback = progress_callback
            self._log_callback = log_callback

            # Update status
            self._progress = ProgressInfo(
                status=RewriteStatus.RUNNING,
                filename=params.input_file,
                output_name=params.output_file,
            )

            # Apply settings to params if not explicitly set
            params_dict = params.to_dict()
            if params_dict.get("base_url") is None:
                params_dict["base_url"] = self._settings.get_api_base_url()
            if params_dict.get("token") is None:
                params_dict["token"] = self._settings.get_api_key()
            if not params_dict.get("rewriter_model"):
                params_dict["rewriter_model"] = self._settings.get_model_name()

            # Start rewrite thread
            self._thread = threading.Thread(
                target=self._run_rewrite,
                args=(params_dict,),
                daemon=True,
            )
            self._thread.start()
            logger.info(f"Started rewrite job: {params.input_file}")
            return True

    def stop_rewrite(self) -> None:
        """Stop the current rewrite job."""
        with self._lock:
            if not self._progress.is_running:
                return

            self._progress.status = RewriteStatus.STOPPING
            self._stop_event.set()
            logger.info("Requested rewrite stop")

    def get_status(self) -> ProgressInfo:
        """Get current job status and progress."""
        with self._lock:
            return ProgressInfo(
                current=self._progress.current,
                total=self._progress.total,
                status=self._progress.status,
                filename=self._progress.filename,
                output_name=self._progress.output_name,
                error_message=self._progress.error_message,
            )

    def get_progress(self) -> tuple[int, int]:
        """Get current progress as (current, total) tuple."""
        with self._lock:
            return self._progress.current, self._progress.total

    def is_running(self) -> bool:
        """Check if a job is currently running."""
        with self._lock:
            return self._progress.is_running

    def wait_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for the current job to complete.

        Args:
            timeout: Maximum time to wait in seconds, None for indefinite

        Returns:
            True if job completed, False if timeout reached
        """
        if self._thread is None:
            return True

        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _run_rewrite(self, params: dict) -> None:
        """
        Internal method to run rewrite in a thread.

        Args:
            params: Parameters dict for rewrite_process
        """
        try:
            # Run the actual rewrite
            success = rewrite_process(
                params=params,
                progress_callback=self._internal_progress_callback,
                stop_event=self._stop_event,
                log_callback=self._internal_log_callback,
                parallel=params.get("parallel", False),
                max_workers=params.get("max_workers"),
            )

            # Update final status
            with self._lock:
                if self._stop_event.is_set():
                    self._progress.status = RewriteStatus.STOPPED
                elif success:
                    self._progress.status = RewriteStatus.COMPLETED
                else:
                    self._progress.status = RewriteStatus.FAILED
                    self._progress.error_message = "Rewrite process failed"

        except Exception as e:
            logger.exception("Rewrite job failed with exception")
            with self._lock:
                self._progress.status = RewriteStatus.FAILED
                self._progress.error_message = str(e)

        finally:
            # Clear thread reference
            with self._lock:
                self._thread = None

    def _internal_progress_callback(self, current: int, total: int) -> None:
        """Internal progress callback that updates state and calls external callback."""
        with self._lock:
            self._progress.current = current
            self._progress.total = total

        if self._progress_callback:
            try:
                self._progress_callback(current, total)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _internal_log_callback(self, message: str) -> None:
        """Internal log callback that calls external callback."""
        if self._log_callback:
            try:
                self._log_callback(message)
            except Exception as e:
                logger.warning(f"Log callback error: {e}")


# Singleton instance for convenience
_rewrite_service: RewriteService | None = None


def get_rewrite_service() -> RewriteService:
    """
    Get the global RewriteService instance.

    Creates a new instance on first call, then caches it.
    Use this for simple use cases where dependency injection is not needed.
    """
    global _rewrite_service
    if _rewrite_service is None:
        _rewrite_service = RewriteService()
    return _rewrite_service
