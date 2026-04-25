"""
Application Service Layer for AI Book Rewriter.

Provides clean interfaces and implementations for:
- RewriteService: High-level rewrite orchestration
- ModelProvider: Model discovery and validation
- PromptService: Prompt catalog management with i18n
- ProgressReporter: Progress and status reporting
"""

from core.services.interfaces import ProgressInfo, RewriteParams, RewriteStatus
from core.services.model_provider import ModelProvider
from core.services.prompt_service import CategoryInfo, PromptInfo, PromptService
from core.services.rewrite_service import RewriteService

__all__ = [
    "CategoryInfo",
    "ModelProvider",
    "ProgressInfo",
    "PromptInfo",
    "PromptService",
    "RewriteParams",
    # Implementations
    "RewriteService",
    # Data classes
    "RewriteStatus",
]
