"""
CLI interface launcher for AI Book Rewriter.

Usage:
    python -m app.cli

This module provides the entry point for running the CLI interface
as a Python module. Currently a placeholder for future implementation.

TODO: Implement CLI interface with:
- Command-line argument parsing
- Input file processing
- Progress output to stdout
- Output file generation
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for 'core' and 'i18n' imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    """Entry point for CLI interface (placeholder)."""
    print("CLI interface is not yet implemented.")
    print("Use one of the following interfaces:")
    print("  python -m app.web  — Web interface")
    print("  python -m app.gui  — GUI interface")
    print("  python main.py --web  — Web via main.py")
    print("  python main.py --gui  — GUI via main.py (default)")
    sys.exit(1)


if __name__ == "__main__":
    main()
