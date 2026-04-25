"""
Local LLM API client.
Uses requests + SSE streaming for compatibility with proxy-specific parameters.
"""

import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.parse

import requests
from urllib3.util.retry import Retry

from core.config import (
    ADAPTIVE_TEMPERATURE_BASE,
    ADAPTIVE_TEMPERATURE_MAX,
    ADAPTIVE_TEMPERATURE_MIN,
    ALLOW_PRIVATE_ENDPOINTS,
    API_TIMEOUT,
    LOCAL_API_BASE_URL,
    LOCAL_API_TOKEN,
    LOCAL_MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    get_timeout,
)
from core.text_engine import count_chars, validate_rewritten_text

logger = logging.getLogger(__name__)


def _validate_base_url(url: str) -> None:
    """Validate API base URL to reduce SSRF risk."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("API base URL must use http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not include credentials")
    if not parsed.hostname:
        raise ValueError("API base URL must include a hostname")
    if ALLOW_PRIVATE_ENDPOINTS:
        return

    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve API base URL hostname '{parsed.hostname}'") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"API base URL resolves to blocked private or link-local address: {ip}")
        if ip == ipaddress.ip_address("169.254.169.254"):
            raise ValueError("API base URL resolves to blocked cloud metadata address")


# Global session for connection pooling
_api_session: requests.Session | None = None
_session_lock = threading.Lock()


def get_api_session() -> requests.Session:
    """Get or create global requests session with connection pooling (thread-safe)."""
    global _api_session
    if _api_session is None:
        with _session_lock:
            if _api_session is None:
                _api_session = requests.Session()
                retry_policy = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[502, 503, 504],
                    allowed_methods=None,
                )
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=retry_policy,
                )
                _api_session.mount("http://", adapter)
                _api_session.mount("https://", adapter)
    return _api_session


def parse_json_response(text: str) -> tuple[str | None, dict | None]:
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
        _validate_base_url(base_url)
        resp = get_api_session().get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", []) if "id" in m]
        return sorted(models) if models else [LOCAL_MODEL_NAME]
    except ValueError as e:
        logger.warning(f"Could not fetch model list: invalid API base URL: {e}")
        return [LOCAL_MODEL_NAME]
    except requests.Timeout as e:
        logger.warning(f"Could not fetch model list: request timed out after {API_TIMEOUT}s: {e}")
        return [LOCAL_MODEL_NAME]
    except requests.ConnectionError as e:
        logger.warning(f"Could not fetch model list: connection failed, check API endpoint availability: {e}")
        return [LOCAL_MODEL_NAME]
    except requests.HTTPError as e:
        logger.warning(f"Could not fetch model list: API returned HTTP error: {e}")
        return [LOCAL_MODEL_NAME]
    except json.JSONDecodeError as e:
        logger.warning(f"Could not fetch model list: API returned invalid JSON: {e}")
        return [LOCAL_MODEL_NAME]
    except requests.RequestException as e:
        logger.warning(f"Could not fetch model list: request failed: {e}")
        return [LOCAL_MODEL_NAME]


def calculate_adaptive_temperature(
    failed_attempts: int,
    quality_metrics: dict[str, float] | None = None,
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
    previous_quality_metrics: dict[str, float] | None = None,
    base_url: str = LOCAL_API_BASE_URL,
    token: str = LOCAL_API_TOKEN,
) -> tuple[str, dict | None] | None:
    """
    Calls the local LLM API with SSE streaming.
    Returns (rewritten_block, context_update) or None on failure.
    """
    try:
        _validate_base_url(base_url)
    except ValueError as e:
        logger.error(f"Invalid API base URL: {e}")
        return None

    adaptive_temp = calculate_adaptive_temperature(failed_attempts, previous_quality_metrics)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    for attempt in range(MAX_RETRIES):
        if stop_event and stop_event.is_set():
            return None

        if attempt > 0:
            adaptive_temp = calculate_adaptive_temperature(failed_attempts + attempt, previous_quality_metrics)

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
            timeout = get_timeout() if base_url == LOCAL_API_BASE_URL else 120
            response = get_api_session().post(
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
                        except json.JSONDecodeError as e:
                            logger.debug(f"{ctx}: Skipping malformed streaming JSON chunk: {e}")

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

        except requests.Timeout as e:
            logger.error(f"{ctx}: API request timed out after {timeout}s; increase API_REWRITE_TIMEOUT or check endpoint latency: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.ConnectionError as e:
            logger.error(f"{ctx}: API connection failed; check base URL, network, or proxy availability: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            logger.error(f"{ctx}: API returned HTTP error {status_code}; check provider status and request parameters: {e}")
            if status_code in {502, 503, 504}:
                wait_time = min(30, RETRY_DELAY_SECONDS * (2**attempt))
                logger.warning(f"{ctx}: transient upstream error - waiting {wait_time}s")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except json.JSONDecodeError as e:
            logger.error(f"{ctx}: API returned invalid streaming JSON; check proxy/provider response format: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.RequestException as e:
            logger.error(f"{ctx}: API request failed; check endpoint configuration and retry policy: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    logger.error("Exhausted all retry attempts")
    return None
