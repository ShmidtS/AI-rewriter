"""
Simple i18n loader.
Usage:
from i18n import tr, set_language, get_supported_languages, get_output_languages
set_language('en')
tr('start') # -> 'Start'
"""
import json
import os
from typing import Any

_TRANSLATIONS: dict = {}
_CURRENT_LANG: str = "ru"
_FALLBACK_LANG: str = "en"
_LANG_DIR: str = os.path.dirname(__file__)

# Cache for loaded configuration
_SUPPORTED_LANGS_CACHE: dict | None = None
_OUTPUT_LANGS_CACHE: list | None = None


def _load_json(filename: str) -> dict:
    """Load a JSON file from the i18n directory."""
    path = os.path.join(_LANG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load(lang: str) -> dict:
    path = os.path.join(_LANG_DIR, f"{lang}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_supported_languages() -> dict:
    """Load supported interface languages from JSON config."""
    global _SUPPORTED_LANGS_CACHE
    if _SUPPORTED_LANGS_CACHE is None:
        config = _load_json("supported_languages.json")
        _SUPPORTED_LANGS_CACHE = config.get("languages", {"ru": "Русский", "en": "English", "zh": "中文"})
    return _SUPPORTED_LANGS_CACHE


def get_supported_languages() -> dict:
    """
    Get dictionary of supported interface languages.
    Returns: {"ru": "Русский", "en": "English", "zh": "中文"}
    """
    return _load_supported_languages()


def get_output_languages() -> list:
    """
    Get list of available output/translation languages.
    Returns: ["Русский", "English", "Español", ...]
    """
    global _OUTPUT_LANGS_CACHE
    if _OUTPUT_LANGS_CACHE is None:
        config = _load_json("output_languages.json")
        _OUTPUT_LANGS_CACHE = config.get("languages", [
            "Русский", "English", "Español", "Français", "Deutsch",
            "Italiano", "Português", "中文", "日本語", "한국어",
            "العربية", "हिन्दी", "Türkçe", "Polski", "Nederlands",
            "Svenska", "Norsk", "Dansk", "Suomi", "Čeština",
            "Slovenčina", "Magyar", "Română", "Български", "Українська",
            "Ελληνικά", "עברית", "ไทย", "Tiếng Việt", "Bahasa Indonesia",
        ])
    return _OUTPUT_LANGS_CACHE


# Legacy: Load SUPPORTED_LANGS at module level for backwards compatibility
SUPPORTED_LANGS = _load_supported_languages()


def set_language(lang: str) -> None:
    global _CURRENT_LANG, _TRANSLATIONS
    _CURRENT_LANG = lang
    fallback = _load(_FALLBACK_LANG)
    current = _load(lang)
    fallback.update(current)
    _TRANSLATIONS = fallback


def tr(key: str, **kwargs: Any) -> str:
    template = _TRANSLATIONS.get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template
    return template


# Load default language on import
set_language(_CURRENT_LANG)
