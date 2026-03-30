"""
Application Service Layer for AI Book Rewriter.

Provides clean interfaces and implementations for:
- RewriteService: High-level rewrite orchestration
- ModelProvider: Model discovery and validation
- PromptService: Prompt catalog management with i18n
- ProgressReporter: Progress and status reporting
"""

from core.services.interfaces import (
    IRewriteService,
    IModelProvider,
    IProgressReporter,
    RewriteStatus,
    ProgressInfo,
    RewriteParams,
)
from core.services.rewrite_service import RewriteService
from core.services.model_provider import ModelProvider
from core.services.prompt_service import PromptService, PromptInfo, CategoryInfo

__all__ = [
    # Interfaces
    "IRewriteService",
    "IModelProvider",
    "IProgressReporter",
    # Data classes
    "RewriteStatus",
    "ProgressInfo",
    "RewriteParams",
    "PromptInfo",
    "CategoryInfo",
    # Implementations
    "RewriteService",
    "ModelProvider",
    "PromptService",
]
