"""
Configuration and constants for AI Book Rewriter.
Loads settings from .env file.

This module provides backward-compatible module-level constants
while internally using the typed Settings from settings.py.
"""

from core.settings import ConnectionProfile, get_settings

# Initialize settings (loads from .env)
_settings = get_settings()

# =============================================================================
# API Connection Settings (backward-compatible)
# =============================================================================

# API base URL - uses profile-specific URL or legacy OPENAI_BASE_URL
LOCAL_API_BASE_URL: str = _settings.get_api_base_url()

# API token - uses profile-specific key or legacy AUTH_TOKEN
LOCAL_API_TOKEN: str = _settings.get_api_key()

# Model name - uses MODEL_NAME or legacy MODEL
LOCAL_MODEL_NAME: str = _settings.get_model_name()

# Context window - uses MODEL_CONTEXT_WINDOW or legacy CONTEXT_WINDOW
LOCAL_CONTEXT_WINDOW: int = _settings.get_context_window()

# Max output tokens - uses MODEL_MAX_OUTPUT_TOKENS or legacy MAX_OUTPUT_TOKENS
LOCAL_MAX_OUTPUT_TOKENS: int = _settings.get_max_output_tokens()

# =============================================================================
# Text Processing Constants
# =============================================================================

# Block target chars - now loaded from REWRITE_BLOCK_TARGET_CHARS
BLOCK_TARGET_CHARS: int = _settings.rewrite_block_target_chars

# Rewrite length ratio limits
MIN_REWRITE_LENGTH_RATIO: float = _settings.rewrite_min_rewrite_length_ratio
MAX_REWRITE_LENGTH_RATIO: float = _settings.rewrite_max_rewrite_length_ratio

# Similarity threshold for duplicate detection
SIMILARITY_THRESHOLD: float = 0.95
MIN_BLOCK_LEN_FACTOR: float = 0.5
MAX_BLOCK_LEN_FACTOR: float = 1.5
SEARCH_RADIUS_FACTOR: float = 0.1

# =============================================================================
# Markers
# =============================================================================

START_MARKER: str = "<|~START_REWRITE~|>"
END_MARKER: str = "<|~END_REWRITE~|>"

# =============================================================================
# Retry / Temperature Settings
# =============================================================================

MAX_RETRIES: int = _settings.rewrite_max_retries
RETRY_DELAY_SECONDS: int = _settings.rewrite_retry_delay_seconds
ADAPTIVE_TEMPERATURE_BASE: float = _settings.model_temperature
ADAPTIVE_TEMPERATURE_MIN: float = 0.5
ADAPTIVE_TEMPERATURE_MAX: float = 1.2

# =============================================================================
# Duplicate Detection
# =============================================================================

MAX_SENTENCE_SIMILARITY_THRESHOLD: float = 0.95
MIN_WORDS_FOR_DUPLICATE_CHECK: int = 10

# =============================================================================
# File Suffixes
# =============================================================================

STATE_SUFFIX: str = "_rewrite_state.json"
INTERMEDIATE_SUFFIX: str = "_intermediate.txt"
FINAL_SUFFIX: str = "_final_rewritten.txt"

# =============================================================================
# Split Priorities (ordered by preference)
# =============================================================================

SPLIT_PRIORITY_ENHANCED: list = [
    "\\n\\n",
    "\\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
]

# =============================================================================
# Default Model Fallback
# =============================================================================

REWRITER_MODEL_DEFAULT: str = LOCAL_MODEL_NAME

# =============================================================================
# Connection Profile Helpers
# =============================================================================


def get_connection_profile() -> ConnectionProfile:
    """Get the current connection profile."""
    return _settings.connection_profile


def is_proxy_mode() -> bool:
    """Check if proxy mode is active."""
    return _settings.is_proxy_mode()


def get_timeout() -> int:
    """Get appropriate timeout based on connection profile."""
    return _settings.get_timeout()


# =============================================================================
# Re-export Settings for direct access
# =============================================================================

__all__ = [
    "ADAPTIVE_TEMPERATURE_BASE",
    "ADAPTIVE_TEMPERATURE_MAX",
    "ADAPTIVE_TEMPERATURE_MIN",
    "BLOCK_TARGET_CHARS",
    "END_MARKER",
    "FINAL_SUFFIX",
    "INTERMEDIATE_SUFFIX",
    # Legacy constants
    "LOCAL_API_BASE_URL",
    "LOCAL_API_TOKEN",
    "LOCAL_CONTEXT_WINDOW",
    "LOCAL_MAX_OUTPUT_TOKENS",
    "LOCAL_MODEL_NAME",
    "MAX_BLOCK_LEN_FACTOR",
    "MAX_RETRIES",
    "MAX_REWRITE_LENGTH_RATIO",
    "MAX_SENTENCE_SIMILARITY_THRESHOLD",
    "MIN_BLOCK_LEN_FACTOR",
    "MIN_REWRITE_LENGTH_RATIO",
    "MIN_WORDS_FOR_DUPLICATE_CHECK",
    "RETRY_DELAY_SECONDS",
    "REWRITER_MODEL_DEFAULT",
    "SEARCH_RADIUS_FACTOR",
    "SIMILARITY_THRESHOLD",
    "SPLIT_PRIORITY_ENHANCED",
    "START_MARKER",
    "STATE_SUFFIX",
    "ConnectionProfile",
    # New helpers
    "get_connection_profile",
    # Settings access
    "get_settings",
    "get_timeout",
    "is_proxy_mode",
]
