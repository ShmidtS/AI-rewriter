"""
Flask web interface for AI Book Rewriter.
Provides web-like UI with real-time SSE progress streaming.
"""
import contextlib
import json
import logging
import os
import queue
import re
import secrets
import threading
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file, session
from werkzeug.utils import secure_filename

from core.config import FINAL_SUFFIX
from core.prompts import get_preset_names
from core.services import ModelProvider, RewriteParams, RewriteService
from core.services.model_provider import get_model_provider
from core.services.prompt_service import get_prompt_service
from core.services.rewrite_service import get_rewrite_service
from core.settings import get_settings, reload_settings
from i18n import get_output_languages, get_supported_languages, set_language, tr

logger = logging.getLogger(__name__)

# Global state (initialized in create_app)
_app_instance: Flask | None = None
_job_lock: threading.Lock | None = None
_event_queues: list = []
_active_job: dict = {}
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
ALLOWED_EXTENSIONS = {'.txt'}
_SAFE_PATH_PARAM_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_upload_filename(filename: str) -> str:
    safe = secure_filename(filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .txt files supported")
    if not safe or safe == extension.lstrip("."):
        safe = f"upload_{uuid.uuid4().hex[:8]}{extension}"
    return safe


def _validate_output_filename(filename: str) -> str:
    if "/" in filename or "\\" in filename:
        raise ValueError("Invalid output filename")
    safe = secure_filename(filename)
    if not safe:
        raise ValueError("Invalid output filename")
    if Path(safe).suffix.lower() != ".txt":
        safe += ".txt"
    return safe


def _validate_int_field(value: Any, field_name: str, min_val: int, max_val: int, default: int) -> int:
    if value in (None, ""):
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name}") from exc
    return max(min_val, min(max_val, parsed))


def _validate_language(lang: str) -> str:
    if not lang:
        return ""
    supported = set(get_output_languages())
    supported.update(get_supported_languages().keys())
    if lang in supported:
        return lang
    return ""


def _validate_safe_path_param(value: str, field_name: str) -> str:
    if not value or not _SAFE_PATH_PARAM_RE.fullmatch(value):
        raise ValueError(f"Invalid {field_name}")
    return value


# Service instances
_rewrite_service: RewriteService | None = None
_model_provider: ModelProvider | None = None


def create_app() -> Flask:
    """
    Application factory for Flask web interface.

    Creates and configures the Flask application instance.
    Routes are registered via @app.route decorators below.
    """
    global _app_instance, _job_lock, _rewrite_service, _model_provider

    # Ensure directories exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Initialize global state
    _job_lock = threading.Lock()

    # Initialize services
    _rewrite_service = get_rewrite_service()
    _model_provider = get_model_provider()

    # Create Flask app
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 # 100 MB
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    _app_instance = app
    return app


# Create app instance for backwards compatibility with imports
# Routes are registered via @app.route decorators below
app = create_app()


def _get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        session["csrf_token"] = token
    return token


def require_csrf(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method == "POST":
            token = request.headers.get("X-CSRF-Token", "")
            if not token or not secrets.compare_digest(token, _get_csrf_token()):
                return jsonify({"error": "Invalid CSRF token"}), 403
        return func(*args, **kwargs)
    return wrapper


@app.route("/api/csrf-token")
def api_csrf_token():
    return jsonify({"csrf_token": _get_csrf_token()})


def _broadcast(event: str, data: dict):
    """Push SSE event to all connected clients."""
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    if _job_lock is None:
        raise RuntimeError("Job lock not initialized")
    with _job_lock:
        for q in list(_event_queues):
            with contextlib.suppress(queue.Full):
                q.put_nowait(msg)


def _progress_cb(current: int, total: int):
    pct = (current / total * 100) if total > 0 else 0
    _broadcast("progress", {"current": current, "total": total, "pct": round(pct, 1)})
    if _job_lock is None:
        raise RuntimeError("Job lock not initialized")
    with _job_lock:
        if _active_job:
            _active_job["current"] = current
            _active_job["total"] = total


def _log_cb(msg: str):
    _broadcast("log", {"msg": msg})


@app.route("/")
def index():
    lang = request.args.get("lang", "ru")
    set_language(lang)
    models = _model_provider.get_available_models() if _model_provider else []
    presets = get_preset_names(lang)
    prompt_svc = get_prompt_service()
    categories = [c.to_dict() for c in prompt_svc.get_categories(lang)]
    return render_template(
        "index.html",
        tr=tr,
        lang=lang,
        supported_langs=get_supported_languages(),
        models=models,
        presets=presets,
        categories=categories,
        output_languages=get_output_languages(),
        active_job=_get_job_status(),
        csrf_token=_get_csrf_token(),
    )


@app.route("/api/start", methods=["POST"])
@require_csrf
def api_start():
    if _rewrite_service is None:
        logger.error("POST /api/start - Service not initialized")
        return jsonify({"error": "Service not initialized"}), 500

    if _rewrite_service.is_running():
        logger.warning("POST /api/start - Job already running")
        return jsonify({"error": "Job already running"}), 400

    uploaded = request.files.get("input_file")
    if not uploaded or not uploaded.filename:
        logger.warning("POST /api/start - No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = _validate_upload_filename(uploaded.filename)
        uploaded.stream.seek(0, os.SEEK_END)
        file_size = uploaded.stream.tell()
        uploaded.stream.seek(0)
        if file_size <= 0:
            raise ValueError("Uploaded file is empty")

        output_file_param = request.form.get("output_file", "").strip()
        if output_file_param:
            output_name = _validate_output_filename(output_file_param)
        else:
            output_name = Path(filename).stem + FINAL_SUFFIX
        max_workers = _validate_int_field(
            request.form.get("parallel_max_workers"),
            "parallel_max_workers",
            1,
            8,
            4,
        )
        language = _validate_language(request.form.get("language", "Русский")) or "Русский"
        output_language = _validate_language(request.form.get("output_language", ""))
    except ValueError as exc:
        logger.warning("POST /api/start - Validation failed: %s", exc)
        return jsonify({"error": str(exc)}), 400

    input_path = UPLOAD_FOLDER / filename
    uploaded.save(str(input_path))
    output_path = OUTPUT_FOLDER / output_name

    params = RewriteParams(
        input_file=str(input_path),
        output_file=str(output_path),
        language=language,
        output_language=output_language,
        style=request.form.get("style", ""),
        goal=request.form.get("goal", ""),
        model=request.form.get("model", ""),
        resume=request.form.get("resume", "true").lower() == "true",
        parallel=request.form.get("parallel_mode", "false").lower() == "true",
        max_workers=max_workers,
        save_interval=1,
        prompt_preset=request.form.get("prompt_preset", "literary"),
    )

    def on_complete():
        _broadcast("done", {"output": output_name})

    started = _rewrite_service.start_rewrite(
        params=params,
        progress_callback=_progress_cb,
        log_callback=_log_cb,
    )

    if started:
        return jsonify({"status": "started", "output": output_name})
    else:
        return jsonify({"error": "Failed to start rewrite"}), 500


@app.route("/api/stop", methods=["POST"])
@require_csrf
def api_stop():
    if _rewrite_service:
        _rewrite_service.stop_rewrite()
    return jsonify({"status": "stopping"})


@app.route("/api/status")
def api_status():
    return jsonify(_get_job_status())


@app.route("/api/models")
def api_models():
    if _model_provider:
        return jsonify(_model_provider.get_available_models())
    return jsonify([])


@app.route("/api/prompts")
def api_prompts():
    """Get all prompts with localized names and descriptions."""
    lang = request.args.get("lang", "en")
    prompt_service = get_prompt_service()
    prompts = prompt_service.get_all_for_api(lang)
    return jsonify(prompts)


@app.route("/api/prompts/<prompt_id>")
def api_prompt_detail(prompt_id: str):
    """Get a specific prompt by ID with localized content."""
    try:
        prompt_id = _validate_safe_path_param(prompt_id, "prompt_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    lang = request.args.get("lang", "en")
    prompt_service = get_prompt_service()
    prompt = prompt_service.get_prompt_for_api(prompt_id, lang)
    if prompt:
        return jsonify(prompt)
    return jsonify({"error": "Prompt not found"}), 404


@app.route("/api/prompts/categories")
def api_prompt_categories():
    """Get all prompt categories with localized names."""
    lang = request.args.get("lang", "en")
    prompt_service = get_prompt_service()
    categories = [c.to_dict() for c in prompt_service.get_categories(lang)]
    return jsonify(categories)


@app.route("/api/download/<filename>")
def api_download(filename):
    safe = secure_filename(filename)
    if not safe or safe != filename:
        return jsonify({"error": "Invalid filename"}), 404
    output_dir = OUTPUT_FOLDER.resolve()
    path = (output_dir / safe).resolve()
    if output_dir not in path.parents and path != output_dir:
        return jsonify({"error": "Invalid filename"}), 404
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


@app.route("/events")
def events():
    """SSE endpoint for real-time progress and logs."""
    q: queue.Queue = queue.Queue(maxsize=200)
    if _job_lock is None:
        raise RuntimeError("Job lock not initialized")
    with _job_lock:
        _event_queues.append(q)

    def generate():
        try:
            # Send current status on connect
            status = _get_job_status()
            yield f"event: status\ndata: {json.dumps(status)}\n\n"
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            if _job_lock is None:
                raise RuntimeError("Job lock not initialized")
            with _job_lock, contextlib.suppress(ValueError):
                _event_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _get_job_status() -> dict:
    if _rewrite_service is None:
        return {
            "running": False,
            "current": 0,
            "total": 0,
            "filename": "",
            "output": "",
        }
    progress = _rewrite_service.get_status()
    return {
        "running": progress.is_running,
        "current": progress.current,
        "total": progress.total,
        "filename": progress.filename,
        "output": progress.output_name,
    }


@app.route("/api/output-languages")
def api_output_languages():
    """Get list of available output/translation languages."""
    return jsonify(get_output_languages())


@app.route("/api/i18n/<lang>.json")
def api_i18n(lang: str):
    """Get translations for frontend JavaScript."""
    try:
        lang = _validate_safe_path_param(lang, "lang")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    from i18n import _load
    translations = _load(lang)
    # Merge with fallback language
    fallback = _load("en")
    fallback.update(translations)
    return jsonify(fallback)


@app.route("/api/settings")
def api_settings_get():
    """Return current non-sensitive settings via JSON."""
    s = get_settings()
    return jsonify({
        "model_name": s.model_name,
        "connection_profile": s.connection_profile.value,
        "ui_lang": s.ui_lang,
        "block_target_chars": s.rewrite_block_target_chars,
        "max_retries": s.rewrite_max_retries,
        "temperature": s.model_temperature,
        "context_window": s.model_context_window,
        "max_output_tokens": s.model_max_output_tokens,
    })


@app.route("/api/settings", methods=["POST"])
@require_csrf
def api_settings_set():
    """Update non-sensitive settings and persist to .env."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Only allow safe, non-sensitive keys to be updated
    allowed_keys = {
        "ui_lang", "block_target_chars", "max_retries",
        "connection_profile", "temperature", "context_window",
        "max_output_tokens",
    }
    to_update = {k: v for k, v in data.items() if k in allowed_keys}
    if not to_update:
        return jsonify({"error": "No updatable keys provided"}), 400

    # Map our API keys to .env variable names
    env_mapping = {
        "ui_lang": "UI_LANG",
        "connection_profile": "CONNECTION_PROFILE",
        "block_target_chars": "REWRITE_BLOCK_TARGET_CHARS",
        "max_retries": "REWRITE_MAX_RETRIES",
        "temperature": "MODEL_TEMPERATURE",
        "context_window": "MODEL_CONTEXT_WINDOW",
        "max_output_tokens": "MODEL_MAX_OUTPUT_TOKENS",
    }

    # Try to update .env file directly
    env_path = Path(".env")
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8")
        # Read existing lines to update in-place
        old_lines: dict[str, str] = {}
        for line in env_content.splitlines():
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key, _, val = stripped.partition("=")
                old_lines[key.strip()] = val.strip()

        updated = False
        for api_key, env_key in env_mapping.items():
            if api_key in to_update:
                old_lines[env_key] = str(to_update[api_key])
                updated = True

        if updated:
            new_lines = []
            for line in env_content.splitlines():
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    line_key = stripped.partition("=")[0].strip()
                    if line_key in old_lines:
                        new_lines.append(f"{line_key}={old_lines[line_key]}")
                        continue
                new_lines.append(line)

            # Append any new keys that weren't already in the file
            written_keys = set()
            for line in env_content.splitlines():
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    written_keys.add(stripped.partition("=")[0].strip())
            for env_key, val in old_lines.items():
                if env_key not in written_keys:
                    new_lines.append(f"{env_key}={val}")

            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # Reload settings to reflect the change
    reload_settings()
    return jsonify({"status": "ok", "updated": list(to_update.keys())})


@app.route("/api/presets/by-category/<category>")
def api_presets_by_category(category: str):
    """Return prompts filtered by category."""
    try:
        category = _validate_safe_path_param(category, "category")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    lang = request.args.get("lang", "en")
    prompt_svc = get_prompt_service()
    prompts = prompt_svc.get_prompts_by_category(category, lang)
    if not prompts:
        return jsonify({"error": f"Category not found: {category}"}), 404
    return jsonify([p.to_dict() for p in prompts])


@app.route("/api/health")
def api_health():
    """Healthcheck: verify API is configured and model list is available."""
    s = get_settings()
    model_list = []
    models_available = False
    if _model_provider:
        try:
            model_list = _model_provider.get_available_models()
            models_available = len(model_list) > 0
        except Exception:
            pass

    return jsonify({
        "status": "ok" if models_available else "degraded",
        "api_configured": bool(s.get_api_base_url() and s.get_api_key()),
        "models_available": models_available,
        "model_count": len(model_list),
        "connection_profile": s.connection_profile.value,
    })


def run_web(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Run the web server. Uses the global app instance."""
    app.run(host=host, port=port, debug=debug, threaded=True)


# NOTE: Direct execution via `python web/app.py` is no longer supported.
# Use one of the following instead:
#   python -m app.web
#   python main.py --web
