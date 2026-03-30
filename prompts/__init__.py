"""
Prompt Catalog Module for AI Book Rewriter.

Provides data-driven prompt management with i18n support.
Catalog is loaded from prompts/catalog.json.
"""

import json
from pathlib import Path
from typing import Optional

# Path to catalog file
CATALOG_PATH = Path(__file__).parent / "catalog.json"

# Cached catalog data
_catalog_cache: Optional[dict] = None


def load_catalog() -> dict:
    """Load and cache the prompt catalog from JSON file."""
    global _catalog_cache
    if _catalog_cache is None:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


def get_catalog_version() -> str:
    """Get catalog version string."""
    catalog = load_catalog()
    return catalog.get("version", "unknown")


def get_all_prompts() -> list[dict]:
    """Get all prompts from catalog."""
    catalog = load_catalog()
    return catalog.get("prompts", [])


def get_prompt_by_id(prompt_id: str) -> Optional[dict]:
    """Get a specific prompt by its ID."""
    prompts = get_all_prompts()
    for prompt in prompts:
        if prompt.get("id") == prompt_id:
            return prompt
    return None


def get_prompts_by_category(category: str) -> list[dict]:
    """Get all prompts in a specific category."""
    prompts = get_all_prompts()
    return [p for p in prompts if p.get("category") == category]


def get_categories() -> list[dict]:
    """Get all available categories."""
    catalog = load_catalog()
    return catalog.get("categories", [])


def get_prompt_name(prompt_id: str, lang: str = "en") -> str:
    """Get localized name for a prompt."""
    prompt = get_prompt_by_id(prompt_id)
    if prompt and "name" in prompt:
        return prompt["name"].get(lang, prompt["name"].get("en", prompt_id))
    return prompt_id


def get_prompt_description(prompt_id: str, lang: str = "en") -> str:
    """Get localized description for a prompt."""
    prompt = get_prompt_by_id(prompt_id)
    if prompt and "description" in prompt:
        return prompt["description"].get(lang, prompt["description"].get("en", ""))
    return ""


def get_prompt_preview(prompt_id: str, lang: str = "en") -> str:
    """Get localized preview text for a prompt."""
    prompt = get_prompt_by_id(prompt_id)
    if prompt and "preview_text" in prompt:
        return prompt["preview_text"].get(lang, prompt["preview_text"].get("en", ""))
    return ""


def get_prompt_template(prompt_id: str) -> Optional[str]:
    """Get the full template for a prompt."""
    prompt = get_prompt_by_id(prompt_id)
    if prompt:
        return prompt.get("full_template")
    return None


def get_all_prompt_names(lang: str = "en") -> dict[str, str]:
    """Get all prompt names as {id: name} dict for given language."""
    prompts = get_all_prompts()
    return {p["id"]: p["name"].get(lang, p["name"].get("en", p["id"])) for p in prompts}


__all__ = [
    "load_catalog",
    "get_catalog_version",
    "get_all_prompts",
    "get_prompt_by_id",
    "get_prompts_by_category",
    "get_categories",
    "get_prompt_name",
    "get_prompt_description",
    "get_prompt_preview",
    "get_prompt_template",
    "get_all_prompt_names",
]
