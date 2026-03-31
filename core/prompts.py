"""
System prompts and prompt construction for the rewriter.
Multiple preset styles supported.

This module provides backward-compatible API while using the new prompt catalog.
"""
import json
from typing import Optional

from core.config import START_MARKER, END_MARKER, MIN_REWRITE_LENGTH_RATIO, MAX_REWRITE_LENGTH_RATIO

# Import from new catalog module for backward compatibility
from prompts import (
    get_all_prompt_names,
    get_prompt_template,
    get_prompt_name,
    get_all_prompts as _get_catalog_prompts,
)


# Lazy-loaded legacy presets for backward compatibility
_SYSTEM_PROMPT_PRESETS_CACHE = None


def _get_legacy_presets():
    """Lazy-load presets in legacy format for backward compatibility."""
    global _SYSTEM_PROMPT_PRESETS_CACHE
    if _SYSTEM_PROMPT_PRESETS_CACHE is None:
        _SYSTEM_PROMPT_PRESETS_CACHE = {}
        for p in _get_catalog_prompts():
            _SYSTEM_PROMPT_PRESETS_CACHE[p["id"]] = {
                "ru": p["name"].get("ru", p["name"].get("en", p["id"])),
                "en": p["name"].get("en", p["id"]),
                "zh": p["name"].get("zh", p["name"].get("en", p["id"])),
                "prompt": p.get("full_template", ""),
            }
    return _SYSTEM_PROMPT_PRESETS_CACHE


# Legacy constant - now a property-like function result
SYSTEM_PROMPT_PRESETS = property(lambda self: _get_legacy_presets())


def get_system_prompt(preset_key: str, min_len: int = 0, max_len: int = 0) -> str:
    """
    Returns the formatted system prompt for the given preset.
    
    Backward-compatible function that now uses the prompt catalog.
    """
    template = get_prompt_template(preset_key)
    if template is None:
        # Fallback to literary if not found
        template = get_prompt_template("literary")
        if template is None:
            return ""
    
    # Format the template with markers and length parameters
    formatted = template.format(
        START_MARKER=START_MARKER,
        END_MARKER=END_MARKER,
        min_len=min_len,
        max_len=max_len,
    )
    
    return formatted


def get_preset_names(lang: str = "en") -> dict:
    """
    Returns {key: display_name} for all presets in the given language.
    
    Backward-compatible function that now uses the prompt catalog.
    """
    return get_all_prompt_names(lang)


def create_rewrite_prompt(
    language: str,
    style: str,
    goal: str,
    block_text: str,
    prev_context: str,
    next_context: str,
    original_len: int,
    global_context=None,
) -> str:
    min_len = int(original_len * MIN_REWRITE_LENGTH_RATIO)
    max_len = int(original_len * MAX_REWRITE_LENGTH_RATIO)

    text_with_markers = f"{START_MARKER}{block_text}{END_MARKER}"

    context_section = ""
    if prev_context or next_context:
        context_section = "\n\nContext for coherence (DO NOT include in output):\n"
        if prev_context:
            context_section += f"Previous text: ...{prev_context}\n"
        if next_context:
            context_section += f"Next text: {next_context}..."

    global_context_section = ""
    if global_context and (global_context.characters or global_context.plot_points or global_context.themes):
        context_json = json.dumps(global_context.to_json(), ensure_ascii=False, indent=2)
        global_context_section = f"""
CURRENT BOOK CONTEXT (for narrative coherence):
{context_json}

Update global_context in your JSON response if new characters/plot/themes appear.
"""

    json_format_section = ""
    if global_context:
        json_format_section = """
RESPONSE FORMAT (JSON only, no other text):
{
    "rewritten_block": "Your rewritten text here...",
    "global_context": {
        "characters": [{"name": "Name", "description": "Brief"}],
        "plot_points": [{"event": "What happened", "significance": "high|medium|low"}],
        "themes": ["theme1"],
        "style_notes": ["observation"]
    }
}

IMPORTANT: Return ONLY valid JSON. Update global_context only if new info appears.
"""

    output_instruction = (
        "7. Output ONLY the rewritten segment - no markers, no explanations"
        if not global_context
        else "7. Return valid JSON with rewritten_block and global_context fields"
    )

    return f"""{global_context_section}LANGUAGE ENFORCEMENT (CRITICAL):
- Output language MUST be: {language}
- NEVER output in any other language
- Every sentence, every word must be in {language}
- Even if you think another language would be better, use {language}

LITERARY EDITORIAL TASK: Professional book rewriting and adaptation

Task: Rewrite the marked text segment

Language: {language}
Style: {style}
Goal: {goal}
Target Length: ~{min_len}-{max_len} characters

CRITICAL REQUIREMENTS:
1. Rewrite ONLY the text between {START_MARKER} and {END_MARKER}
2. Preserve core meaning and all key information
3. Use semantic transformation - different expression, same meaning
4. Ensure lexical diversity - avoid repeating words from original
5. Maintain smooth transitions with surrounding context (if provided)
6. Do NOT repeat any sentences from the context provided below
{output_instruction}

Text to rewrite:
{text_with_markers}{context_section}{json_format_section}"""
