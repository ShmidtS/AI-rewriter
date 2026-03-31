"""
Flask web interface for AI Book Rewriter.
Provides web-like UI with real-time SSE progress streaming.
"""
import os
import json
import threading
import queue
import logging
import uuid
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, Response,
    send_file, redirect, url_for
)
from werkzeug.utils import secure_filename

from core.services import RewriteService, ModelProvider, RewriteParams, RewriteStatus
from core.services.rewrite_service import get_rewrite_service
from core.services.model_provider import get_model_provider
from core.prompts import get_preset_names
from core.config import FINAL_SUFFIX
from i18n import tr, set_language, get_supported_languages, get_output_languages
from core.services.prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

# Global state (initialized in create_app)
_app_instance: Flask | None = None
_job_lock: threading.Lock | None = None
_event_queues: list = []
_active_job: dict = {}
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")

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

    _app_instance = app
    return app


# Create app instance for backwards compatibility with imports
# Routes are registered via @app.route decorators below
app = create_app()


def _broadcast(event: str, data: dict):
    """Push SSE event to all connected clients."""
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _job_lock:
        for q in list(_event_queues):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


def _progress_cb(current: int, total: int):
    pct = (current / total * 100) if total > 0 else 0
    _broadcast("progress", {"current": current, "total": total, "pct": round(pct, 1)})
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
    return render_template(
        "index.html",
        tr=tr,
        lang=lang,
        supported_langs=get_supported_languages(),
        models=models,
        presets=presets,
        output_languages=get_output_languages(),
        active_job=_get_job_status(),
    )


@app.route("/api/start", methods=["POST"])
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

    # Check original filename for .txt extension BEFORE secure_filename
    # (secure_filename strips non-ASCII chars, turning "файл.txt" into "txt")
    original_filename = uploaded.filename
    if not original_filename.lower().endswith(".txt"):
        logger.warning(f"POST /api/start - Rejected: '{original_filename}' is not .txt")
        return jsonify({"error": "Only .txt files supported"}), 400

    # Generate safe filename - use secure_filename, but if it strips everything,
    # fall back to a UUID-based name
    filename = secure_filename(original_filename)
    if not filename or filename == "txt":
        # secure_filename stripped all non-ASCII chars, generate a safe name
        filename = f"upload_{uuid.uuid4().hex[:8]}.txt"

    input_path = UPLOAD_FOLDER / filename
    uploaded.save(str(input_path))
    
    # Use user-specified output filename or auto-generate
    output_file_param = request.form.get("output_file", "").strip()
    if output_file_param:
        # Sanitize and use user-specified name
        output_name = secure_filename(output_file_param)
        if not output_name.endswith(".txt"):
            output_name += ".txt"
    else:
        output_name = Path(filename).stem + FINAL_SUFFIX
    output_path = OUTPUT_FOLDER / output_name
    
    params = RewriteParams(
        input_file=str(input_path),
        output_file=str(output_path),
        language=request.form.get("language", "Русский"),
        style=request.form.get("style", ""),
        goal=request.form.get("goal", ""),
        model=request.form.get("model", ""),
        resume=request.form.get("resume", "true").lower() == "true",
        parallel=request.form.get("parallel_mode", "false").lower() == "true",
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
    path = OUTPUT_FOLDER / safe
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


@app.route("/events")
def events():
    """SSE endpoint for real-time progress and logs."""
    q: queue.Queue = queue.Queue(maxsize=200)
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
            with _job_lock:
                try:
                    _event_queues.remove(q)
                except ValueError:
                    pass

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
    from i18n import _load
    translations = _load(lang)
    # Merge with fallback language
    fallback = _load("en")
    fallback.update(translations)
    return jsonify(fallback)


def run_web(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Run the web server. Uses the global app instance."""
    app.run(host=host, port=port, debug=debug, threaded=True)


# NOTE: Direct execution via `python web/app.py` is no longer supported.
# Use one of the following instead:
#   python -m app.web
#   python main.py --web
