"""
GUI interface launcher for AI Book Rewriter.

Usage:
    python -m app.gui

This module provides the entry point for running the GUI interface
as a Python module, ensuring proper import paths are set up.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for 'core' and 'i18n' imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    """Entry point for GUI interface."""
    from gui.app import run_gui
    run_gui()


if __name__ == "__main__":
    main()
