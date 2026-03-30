"""
Prompt Service for AI Book Rewriter.

Provides high-level API for prompt management with i18n support.
Integrates with the prompts catalog and core.prompts module.
"""

import logging
from typing import Optional

from prompts import (
    get_all_prompts,
    get_prompt_by_id,
    get_prompts_by_category,
    get_categories,
    get_prompt_name,
    get_prompt_description,
    get_prompt_preview,
    get_prompt_template,
    get_all_prompt_names,
)
from core.config import START_MARKER, END_MARKER

logger = logging.getLogger(__name__)


class PromptInfo:
    """Data class for prompt information with i18n support."""
    
    def __init__(self, prompt_data: dict, lang: str = "en"):
        self.id: str = prompt_data.get("id", "")
        self.name: str = get_prompt_name(self.id, lang)
        self.description: str = get_prompt_description(self.id, lang)
        self.preview: str = get_prompt_preview(self.id, lang)
        self.category: str = prompt_data.get("category", "")
        self.template: Optional[str] = prompt_data.get("full_template")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "preview": self.preview,
            "category": self.category,
        }


class CategoryInfo:
    """Data class for category information with i18n support."""
    
    def __init__(self, category_data: dict, lang: str = "en"):
        self.id: str = category_data.get("id", "")
        names = category_data.get("name", {})
        self.name: str = names.get(lang, names.get("en", self.id))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
        }


class PromptService:
    """
    Service for managing prompts with i18n support.
    
    Provides methods to retrieve, filter, and render prompts.
    """
    
    def __init__(self, default_lang: str = "en"):
        self._default_lang = default_lang
    
    def get_all_prompts(self, lang: Optional[str] = None) -> list[PromptInfo]:
        """Get all prompts with localized names and descriptions."""
        lang = lang or self._default_lang
        prompts = get_all_prompts()
        return [PromptInfo(p, lang) for p in prompts]
    
    def get_prompt_by_id(self, prompt_id: str, lang: Optional[str] = None) -> Optional[PromptInfo]:
        """Get a specific prompt by ID with localized content."""
        lang = lang or self._default_lang
        prompt_data = get_prompt_by_id(prompt_id)
        if prompt_data:
            return PromptInfo(prompt_data, lang)
        return None
    
    def get_prompts_by_category(self, category: str, lang: Optional[str] = None) -> list[PromptInfo]:
        """Get all prompts in a specific category."""
        lang = lang or self._default_lang
        prompts = get_prompts_by_category(category)
        return [PromptInfo(p, lang) for p in prompts]
    
    def get_categories(self, lang: Optional[str] = None) -> list[CategoryInfo]:
        """Get all available categories with localized names."""
        lang = lang or self._default_lang
        categories = get_categories()
        return [CategoryInfo(c, lang) for c in categories]
    
    def get_prompt_names(self, lang: Optional[str] = None) -> dict[str, str]:
        """Get all prompt names as {id: name} dict."""
        lang = lang or self._default_lang
        return get_all_prompt_names(lang)
    
    def render_prompt(
        self,
        prompt_id: str,
        min_len: int = 0,
        max_len: int = 0,
    ) -> Optional[str]:
        """
        Render a prompt template with parameters.
        
        Args:
            prompt_id: The prompt ID to render
            min_len: Minimum target length for rewritten text
            max_len: Maximum target length for rewritten text
        
        Returns:
            Formatted prompt string or None if not found
        """
        template = get_prompt_template(prompt_id)
        if template is None:
            logger.warning(f"Prompt not found: {prompt_id}")
            return None
        
        # Format the template with markers and length parameters
        formatted = template.format(
            START_MARKER=START_MARKER,
            END_MARKER=END_MARKER,
        )
        
        # Replace length placeholders (they use double braces in template)
        formatted = formatted.replace("{min_len}", str(min_len))
        formatted = formatted.replace("{max_len}", str(max_len))
        
        return formatted
    
    def get_prompt_for_api(
        self,
        prompt_id: str,
        lang: Optional[str] = None
    ) -> Optional[dict]:
        """
        Get prompt data formatted for API response.
        
        Returns dict with id, name, description, preview, category.
        """
        prompt_info = self.get_prompt_by_id(prompt_id, lang)
        if prompt_info:
            return prompt_info.to_dict()
        return None
    
    def get_all_for_api(self, lang: Optional[str] = None) -> list[dict]:
        """Get all prompts formatted for API response."""
        return [p.to_dict() for p in self.get_all_prompts(lang)]


# Singleton instance
_prompt_service: Optional[PromptService] = None


def get_prompt_service() -> PromptService:
    """Get or create the singleton PromptService instance."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service
