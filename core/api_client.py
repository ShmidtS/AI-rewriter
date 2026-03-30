"""
Local LLM API client.
Uses requests + SSE streaming for compatibility with proxy-specific parameters.
"""
import json
import time
import logging
import threading
from typing import Optional, Tuple, Dict

import requests

from core.config import (
    LOCAL_API_BASE_URL,
    LOCAL_API_TOKEN,
    LOCAL_MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    ADAPTIVE_TEMPERATURE_BASE,
    ADAPTIVE_TEMPERATURE_MIN,
    ADAPTIVE_TEMPERATURE_MAX,
    get_timeout,
    is_proxy_mode,
)
from core.settings import get_settings
from core.text_engine import validate_rewritten_text, count_chars
from core.prompts import create_rewrite_prompt

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Parse API response: JSON with rewritten_block/global_context, or plain text."""
    import re
    if not text:
        return None, None
    text = text.strip()
    try:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return data.get("rewritten_block"), data.get("global_context")
    except json.JSONDecodeError:
        return text, None


def list_available_models(base_url: str = LOCAL_API_BASE_URL, token: str = LOCAL_API_TOKEN):
    """Fetch model list from the proxy /v1/models endpoint."""
    from core.config import LOCAL_MODEL_NAME
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", []) if "id" in m]
        return sorted(models) if models else [LOCAL_MODEL_NAME]
    except Exception as e:
        logger.warning(f"Could not fetch model list: {e}")
        return [LOCAL_MODEL_NAME]


def calculate_adaptive_temperature(
    failed_attempts: int,
    quality_metrics: Optional[Dict[str, float]] = None,
) -> float:
    attempt_factor = min(failed_attempts * 0.05, 0.2)
    similarity_factor = 0.0
    if quality_metrics and quality_metrics.get("similarity", 0) > 0.85:
        similarity_factor = (quality_metrics["similarity"] - 0.85) * 0.4
    diversity_factor = 0.0
    if quality_metrics and quality_metrics.get("diversity", 1) < 0.15:
        diversity_factor = (0.15 - quality_metrics["diversity"]) * 0.3
    final_temp = ADAPTIVE_TEMPERATURE_BASE + attempt_factor + similarity_factor + diversity_factor
    return max(ADAPTIVE_TEMPERATURE_MIN, min(ADAPTIVE_TEMPERATURE_MAX, final_temp))


def call_local_rewrite_api(
    system_instruction: str,
    user_content: str,
    model_name: str,
    orig_len: int,
    original: str,
    prev_block: str,
    next_block: str,
    stop_event: threading.Event,
    global_context,
    failed_attempts: int = 0,
    previous_quality_metrics: Optional[Dict[str, float]] = None,
    base_url: str = LOCAL_API_BASE_URL,
    token: str = LOCAL_API_TOKEN,
) -> Optional[Tuple[str, Optional[Dict]]]:
    """
    Calls the local LLM API with SSE streaming.
    Returns (rewritten_block, context_update) or None on failure.
    """
    adaptive_temp = calculate_adaptive_temperature(failed_attempts, previous_quality_metrics)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    for attempt in range(MAX_RETRIES):
        if stop_event and stop_event.is_set():
            return None

        if attempt > 0:
            adaptive_temp = calculate_adaptive_temperature(
                failed_attempts + attempt, previous_quality_metrics
            )

        ctx = f"Attempt {attempt + 1}/{MAX_RETRIES}"
        logger.info(f"API call: {ctx}")

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": adaptive_temp,
            "top_p": 0.95,
            "max_tokens": LOCAL_MAX_OUTPUT_TOKENS,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": 0,
        }

        try:
            # Use profile-specific timeout
            timeout = get_timeout() if base_url == LOCAL_API_BASE_URL else 300
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            )
            response.raise_for_status()

            text = ""
            for line in response.iter_lines():
                if stop_event and stop_event.is_set():
                    return None
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                text += delta["content"]
                        except Exception:
                            pass

            if not text:
                logger.error(f"{ctx}: Empty response")
                continue

            rewritten_block, context_update = parse_json_response(text)
            if not rewritten_block:
                logger.warning(f"{ctx}: Empty rewritten_block")
                continue

            is_valid, validation_error, quality_metrics = validate_rewritten_text(
                rewritten_block, original, orig_len, prev_block, next_block, ctx
            )

            if is_valid:
                logger.info(f"{ctx}: OK ({count_chars(rewritten_block)} chars)")
                return rewritten_block, context_update
            else:
                logger.warning(f"{ctx}: {validation_error}")
                if quality_metrics:
                    previous_quality_metrics = quality_metrics

        except Exception as e:
            error_msg = str(e)
            logger.error(f"{ctx}: API error: {error_msg}")
            if "502" in error_msg or "Bad Gateway" in error_msg:
                wait_time = min(30, RETRY_DELAY_SECONDS * (2 ** attempt))
                logger.warning(f"{ctx}: 502 - waiting {wait_time}s")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    logger.error("Exhausted all retry attempts")
    return None
