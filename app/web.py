"""
Web interface launcher for AI Book Rewriter.

Usage:
    python -m app.web
    python -m app.web --host 0.0.0.0 --port 5000
    python -m app.web --debug

This module provides the entry point for running the web interface
as a Python module, ensuring proper import paths are set up.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for 'core' and 'i18n' imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    """Entry point for web interface."""
    import argparse

    from web.app import run_web

    parser = argparse.ArgumentParser(description="AI Book Rewriter - Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting web interface at http://{args.host}:{args.port}")
    run_web(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
