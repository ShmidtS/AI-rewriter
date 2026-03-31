"""
Entry point for AI Book Rewriter.

Usage:
python main.py                                  # launches web interface (default)
python main.py --gui                            # launches tkinter GUI
python main.py --web --host 0.0.0.0 --port 5000 # web with custom host/port
python main.py --cli --input book.txt --output out.txt   # CLI rewriting
python main.py --cli --input book.txt --output out.txt \
    --parallel --max-workers 8                  # CLI parallel mode

Alternative entry points:
python -m app.gui                               # GUI via module
python -m app.web                               # Web via module
python -m app.cli --input book.txt --output out.txt   # CLI via module
"""
import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = RotatingFileHandler(
        "rewriter.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="AI Book Rewriter",
        epilog="Use 'python -m app.{web|gui|cli}' for module-style execution."
    )
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--cli", action="store_true", help="Launch CLI interface")
    parser.add_argument("--host", default="127.0.0.1", help="Web host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Web port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    # Use parse_known_args so extra CLI args (--input, --parallel, etc.) pass through
    args, remaining = parser.parse_known_args()

    # Determine which interface to launch
    if args.cli:
        from app.cli import main as cli_main
        # Replace sys.argv so CLI's argparse sees only its own args (without --cli)
        sys.argv = [sys.argv[0]] + remaining
        cli_main()
    elif args.web:
        from web.app import run_web
        print(f"Starting web interface at http://{args.host}:{args.port}")
        run_web(host=args.host, port=args.port, debug=args.debug)
    else:
        # Default to GUI (or --gui explicitly)
        from gui.app import run_gui
        run_gui()


if __name__ == "__main__":
    main()
