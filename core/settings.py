"""
Typed configuration using Pydantic Settings.
Loads settings from .env file with validation and type safety.
"""

from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConnectionProfile(str, Enum):
    """Connection profile for API access."""

    DIRECT = "direct"  # Direct API connection (e.g., OpenAI, Google)
    PROXY = "proxy"  # Through local proxy (E:\456\LLM-API-Key-Proxy)
    AUTO = "auto"  # Auto-detect: try proxy first, fallback to direct


class Settings(BaseSettings):
    """
    Main application settings.

    Supports three connection profiles:
    - direct: Use DIRECT_* settings for API connection
    - proxy: Use PROXY_* settings through local proxy
    - auto: Try proxy first, fallback to direct

    All settings are loaded from .env file with appropriate prefixes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        nested_model_default_factor=0,  # type: ignore[typeddict-unknown-key]
    )

    # =========================================================================
    # Connection Profile
    # =========================================================================
    connection_profile: ConnectionProfile = Field(
        default=ConnectionProfile.PROXY,
        description="Connection profile: direct, proxy, auto",
    )

    # =========================================================================
    # Proxy Settings (PROXY_* prefix)
    # =========================================================================
    proxy_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        alias="PROXY_BASE_URL",
        description="Proxy base URL (OpenAI-compatible endpoint)",
    )
    proxy_api_key: str = Field(
        default="",
        alias="PROXY_API_KEY",
        description="API key for proxy authentication",
    )
    proxy_timeout_connect: int = Field(
        default=30,
        ge=5,
        le=120,
        alias="PROXY_TIMEOUT_CONNECT",
        description="Connection timeout in seconds",
    )
    proxy_timeout_read_streaming: int = Field(
        default=300,
        ge=60,
        le=1800,
        alias="PROXY_TIMEOUT_READ_STREAMING",
        description="Read timeout for streaming responses",
    )
    proxy_healthcheck_enabled: bool = Field(
        default=True,
        alias="PROXY_HEALTHCHECK_ENABLED",
        description="Enable proxy healthcheck before requests",
    )

    # =========================================================================
    # Direct API Settings (DIRECT_* prefix)
    # =========================================================================
    direct_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="DIRECT_BASE_URL",
        description="Direct API base URL",
    )
    direct_api_key: str = Field(
        default="",
        alias="DIRECT_API_KEY",
        description="Direct API key",
    )

    # =========================================================================
    # Model Settings (MODEL_* prefix)
    # =========================================================================
    model_name: str = Field(
        default="gemini/gemini-2.5-flash",
        alias="MODEL_NAME",
        description="Model name in provider/model format",
    )
    model_context_window: int = Field(
        default=1_000_000,
        ge=1000,
        alias="MODEL_CONTEXT_WINDOW",
        description="Model context window size",
    )
    model_max_output_tokens: int = Field(
        default=32768,
        ge=100,
        alias="MODEL_MAX_OUTPUT_TOKENS",
        description="Maximum output tokens per request",
    )
    model_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        alias="MODEL_TEMPERATURE",
        description="Default temperature for generation",
    )

    # =========================================================================
    # Rewrite Settings (REWRITE_* prefix)
    # =========================================================================
    rewrite_block_target_chars: int = Field(
        default=15000,
        ge=1000,
        alias="REWRITE_BLOCK_TARGET_CHARS",
        description="Target block size in characters",
    )
    rewrite_min_rewrite_length_ratio: float = Field(
        default=0.40,
        ge=0.1,
        le=1.0,
        alias="REWRITE_MIN_REWRITE_LENGTH_RATIO",
        description="Minimum rewrite length ratio",
    )
    rewrite_max_rewrite_length_ratio: float = Field(
        default=1.6,
        ge=0.5,
        le=3.0,
        alias="REWRITE_MAX_REWRITE_LENGTH_RATIO",
        description="Maximum rewrite length ratio",
    )
    rewrite_max_retries: int = Field(
        default=20,
        ge=1,
        le=100,
        alias="REWRITE_MAX_RETRIES",
        description="Maximum retries per block",
    )
    rewrite_retry_delay_seconds: int = Field(
        default=2,
        ge=1,
        le=60,
        alias="REWRITE_RETRY_DELAY_SECONDS",
        description="Delay between retries in seconds",
    )

    # =========================================================================
    # UI Settings (UI_* prefix)
    # =========================================================================
    ui_lang: str = Field(
        default="ru",
        alias="UI_LANG",
        description="UI language: ru, en, zh",
    )

    # =========================================================================
    # Legacy Variables (backward compatibility)
    # =========================================================================
    # These are read without prefix for backward compatibility
    api_key: str | None = Field(
        default=None,
        alias="AUTH_TOKEN",
        description="Legacy: API key (use PROXY_API_KEY or DIRECT_API_KEY)",
    )
    api_base_url: str | None = Field(
        default=None,
        alias="OPENAI_BASE_URL",
        description="Legacy: API base URL (use PROXY_BASE_URL or DIRECT_BASE_URL)",
    )
    model_name_legacy: str | None = Field(
        default=None,
        alias="MODEL",
        description="Legacy: Model name (use MODEL_NAME)",
    )
    context_window_legacy: int | None = Field(
        default=None,
        alias="CONTEXT_WINDOW",
        description="Legacy: Context window (use MODEL_CONTEXT_WINDOW)",
    )
    max_output_tokens_legacy: int | None = Field(
        default=None,
        alias="MAX_OUTPUT_TOKENS",
        description="Legacy: Max output tokens (use MODEL_MAX_OUTPUT_TOKENS)",
    )
    block_target_chars_legacy: int | None = Field(
        default=None,
        alias="BLOCK_TARGET_CHARS",
        description="Legacy: Block target chars (use REWRITE_BLOCK_TARGET_CHARS)",
    )

    @field_validator("ui_lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        allowed = {"ru", "en", "zh"}
        if v.lower() not in allowed:
            raise ValueError(f"UI_LANG must be one of: {allowed}")
        return v.lower()

    @model_validator(mode="after")
    def apply_legacy_defaults(self) -> "Settings":
        """Apply legacy values to new fields if new fields are not set."""
        # Model name
        if self.model_name == "gemini/gemini-2.5-flash" and self.model_name_legacy:
            self.model_name = self.model_name_legacy

        # Context window
        if self.model_context_window == 1_000_000 and self.context_window_legacy:
            self.model_context_window = self.context_window_legacy

        # Max output tokens
        if self.model_max_output_tokens == 32768 and self.max_output_tokens_legacy:
            self.model_max_output_tokens = self.max_output_tokens_legacy

        # Block target chars
        if self.rewrite_block_target_chars == 15000 and self.block_target_chars_legacy:
            self.rewrite_block_target_chars = self.block_target_chars_legacy

        return self

    def get_api_base_url(self) -> str:
        """
        Get the effective API base URL based on connection profile.

        Priority:
        1. Profile-specific URL (proxy_base_url or direct_base_url)
        2. Legacy OPENAI_BASE_URL
        3. Profile default
        """
        if self.connection_profile == ConnectionProfile.DIRECT:
            url = self.direct_base_url
        else:  # PROXY or AUTO
            url = self.proxy_base_url

        # Fallback to legacy
        if not url and self.api_base_url:
            url = self.api_base_url

        return url or "http://127.0.0.1:8000/v1"

    def get_api_key(self) -> str:
        """
        Get the effective API key based on connection profile.

        Priority:
        1. Profile-specific key (proxy_api_key or direct_api_key)
        2. Legacy AUTH_TOKEN
        3. Empty string (will fail API calls)
        """
        if self.connection_profile == ConnectionProfile.DIRECT:
            key = self.direct_api_key
        else:  # PROXY or AUTO
            key = self.proxy_api_key

        # Fallback to legacy
        if not key and self.api_key:
            key = self.api_key

        return key or ""

    def get_model_name(self) -> str:
        """Get model name with legacy fallback applied."""
        return self.model_name

    def get_context_window(self) -> int:
        """Get context window with legacy fallback applied."""
        return self.model_context_window

    def get_max_output_tokens(self) -> int:
        """Get max output tokens with legacy fallback applied."""
        return self.model_max_output_tokens

    def get_timeout(self) -> int:
        """Get appropriate timeout based on connection profile."""
        if self.connection_profile in (ConnectionProfile.PROXY, ConnectionProfile.AUTO):
            return self.proxy_timeout_read_streaming
        return 300  # Default 5 minutes for direct

    def is_proxy_mode(self) -> bool:
        """Check if proxy mode is active."""
        return self.connection_profile in (ConnectionProfile.PROXY, ConnectionProfile.AUTO)


# Global settings instance (singleton pattern)
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Loads from .env file on first call, then caches the result.
    Use this function instead of direct Settings() instantiation
    to ensure consistent configuration across the application.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from .env file.

    Useful for testing or when .env file changes during runtime.
    """
    global _settings
    _settings = Settings()
    return _settings
