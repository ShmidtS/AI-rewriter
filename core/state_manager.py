"""
Atomic state and intermediate file persistence.
"""

import json
import logging
import os
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

BlockStatus = Literal["pending", "processed", "failed", "skipped"]


class BlockState(TypedDict, total=False):
    index: int
    original_text: str
    rewritten_text: str
    status: BlockStatus
    block_index: int
    start_char_index: int
    end_char_index: int
    original_char_length: int
    processed: bool
    failed_attempts: int
    last_quality_metrics: dict[str, Any] | None


class RewriteMetadata(TypedDict, total=False):
    processed_block_index: int
    total_blocks: int
    timestamp: float
    global_context: dict[str, Any]


class RewriteState(TypedDict, total=False):
    blocks: list[BlockState]
    params: dict[str, Any]
    metadata: RewriteMetadata
    output_path: str
    processed_block_index: int
    original_blocks_data: list[BlockState]
    total_blocks: int
    timestamp: float
    global_context: dict[str, Any]


class StateManager:
    def __init__(self, filename: str, data: RewriteState | None = None, save_interval: int = 1) -> None:
        self.filename = filename
        self.data = data or {}
        self.save_interval = save_interval
        self._dirty: bool = data is not None
        self._chunks_since_save = save_interval if data is not None else 0

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def bind_state(self, data: RewriteState) -> None:
        self.data = data
        self._dirty = False
        self._chunks_since_save = 0

    def mark_dirty(self, processed_chunks: int = 0) -> None:
        self._dirty = True
        self._chunks_since_save += processed_chunks

    def set_state(self, data: RewriteState, processed_chunks: int = 0) -> None:
        self.data = data
        self.mark_dirty(processed_chunks)

    def save_state(self, force: bool = False) -> bool:
        if not self._dirty:
            return False
        if not force:
            if self.save_interval <= 0:
                return False
            if self._chunks_since_save < self.save_interval:
                return False
        _write_state_file(self.filename, self.data)
        self._dirty = False
        self._chunks_since_save = 0
        return True

    def save_if_dirty(self, force: bool = False) -> bool:
        return self.save_state(force=force)


def _write_state_file(filename: str, data: RewriteState) -> None:
    temp_file = filename + ".tmp"
    try:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, filename)
        logger.debug(f"State saved: {filename}")
    except OSError as e:
        logger.error(f"Error saving state to {filename}: {e}", exc_info=True)
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            logger.debug(f"Could not remove temp state file: {temp_file}", exc_info=True)
        raise


def save_state(filename: str, data: RewriteState, force: bool = False, save_interval: int = 1) -> bool:
    manager = StateManager(filename, data, save_interval)
    return manager.save_state(force=force)


def load_state(filename: str) -> RewriteState | None:
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, encoding="utf-8") as f:
            state: RewriteState = json.load(f)
        required = {"processed_block_index", "original_blocks_data", "total_blocks"}
        if not required.issubset(state.keys()):
            logger.warning(f"State file {filename} has incomplete format")
            return None
        return state
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in {filename}: {e}")
        return None
    except OSError as e:
        logger.error(f"Error loading state from {filename}: {e}", exc_info=True)
        return None


def save_intermediate(filename: str, content: str, context: str = "") -> None:
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_file, filename)
        logger.debug(f"{context}: Intermediate saved: {filename}")
    except OSError as e:
        logger.error(f"{context}: Error saving {filename}: {e}", exc_info=True)
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            logger.debug(f"Could not remove temp intermediate file: {temp_file}", exc_info=True)
        raise
