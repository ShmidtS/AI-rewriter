"""
CLI interface for AI Book Rewriter.

Usage:
    python main.py --cli --input file.txt --output result.txt
    python main.py --cli --input file.txt --output result.txt --parallel --max-workers 8
    python -m app.cli --input file.txt --output result.txt

Full options:
    --input FILE          Input text file (required)
    --output FILE         Output file (required)
    --language LANG       Target language (default: English)
    --style TEXT          Style description (default: literary)
    --goal TEXT           Rewrite goal (default: improve quality)
    --model NAME          Model name (default: from settings)
    --preset PRESET       Prompt preset (default: literary)
    --parallel            Enable parallel rewriting mode
    --max-workers N       Number of parallel workers (default: 10)
    --output-language LANG Alias for --language
    --verbose             Enable verbose logging to console
    --quiet               Suppress all progress output
    --list-presets        Show all available prompt presets and exit
    --no-resume           Start fresh, ignore previous session state
    --save-interval N     Save state every N blocks (default: 1)
"""
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core.config import LOCAL_MODEL_NAME
from core.prompts import get_all_prompt_names
from core.rewriter import rewrite_process


def main():
    """Entry point for the CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ai-rewriter-cli",
        description="AI Book Rewriter -- command-line interface",
        epilog="Examples:\n"
               "  python main.py --cli --input book.txt --output out.txt\n"
               "  python main.py --cli --input book.txt --output out.txt --parallel --max-workers 8\n"
               "  python main.py --cli --input book.txt --output out.txt --preset literary-translation --language ru\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # File arguments (required unless --list-presets)
    parser.add_argument("--input", default=None, help="Input text file path")
    parser.add_argument("--output", default=None, help="Output file path")

    # Rewrite parameters
    parser.add_argument(
        "--language", default="English",
        help="Target output language (default: English)",
    )
    parser.add_argument(
        "--style", default="literary",
        help="Style description (default: literary)",
    )
    parser.add_argument(
        "--goal", default="improve quality",
        help="Rewrite goal (default: improve quality)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: from settings)",
    )
    parser.add_argument(
        "--preset", default="literary",
        help="Prompt preset (default: literary). Use --list-presets to see all.",
    )

    # Parallel mode
    parser.add_argument(
        "--parallel", action="store_true",
        help="Enable parallel rewriting mode (faster, no cross-block context)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Number of parallel workers (default: 10)",
    )

    # Output language alias
    parser.add_argument(
        "--output-language", default=None,
        help="Alias for --language (useful with translation presets)",
    )

    # Preset listing
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Show all available prompt presets and exit",
    )

    # Session control
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignore previous session state",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1,
        help="Save state every N blocks (default: 1)",
    )

    # Logging / verbosity
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging to console",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress all progress output",
    )

    args = parser.parse_args()

    # --list-presets
    if args.list_presets:
        presets = get_all_prompt_names("en")
        print("Available prompt presets:")
        for pid, name in presets.items():
            print(f"  {pid:30s} {name}")
        return

    # Validate required --input / --output
    if not args.input or not args.output:
        parser.error("--input and --output are required")

    # Resolve language
    language = args.output_language if args.output_language else args.language

    # Validate input file
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate preset
    presets = get_all_prompt_names("en")
    if args.preset not in presets:
        print(
            f"Error: Unknown preset '{args.preset}'. Available: {', '.join(sorted(presets.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build params
    model_name = args.model if args.model else LOCAL_MODEL_NAME
    params = {
        "input_file": str(input_path),
        "output_file": args.output,
        "language": language,
        "style": args.style,
        "goal": args.goal,
        "rewriter_model": model_name,
        "resume": not args.no_resume,
        "save_interval": args.save_interval,
        "prompt_preset": args.preset,
    }

    # Setup verbose logging
    if args.verbose:
        logger = logging.getLogger("core")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(ch)

    def _log_callback(msg):
        if args.verbose:
            print(f"  [LOG] {msg}")

    # Print header
    if not args.quiet:
        mode_label = "PARALLEL" if args.parallel else "SEQUENTIAL"
        workers_info = f", workers={args.max_workers}" if args.parallel else ""
        print(f"AI Book Rewriter -- Mode: {mode_label}{workers_info}")
        print(f"Input:  {input_path}")
        print(f"Output: {args.output}")
        print(f"Language: {language} | Style: {args.style} | Preset: {args.preset}")
        print(f"Model:  {model_name}")
        print()

    # Progress callback with tqdm or text fallback
    state: dict[str, Any] = {"total": 0, "last_done": 0, "last_pct": -1, "pbar": None}

    def _progress(current, total):
        if args.quiet:
            return

        if state["total"] != total:
            state["total"] = total
            if HAS_TQDM:
                state["pbar"] = tqdm(total=total, desc="Rewriting", unit="block")
                state["last_done"] = 0

        if HAS_TQDM and state["pbar"] is not None:
            delta = current - state["last_done"]
            if delta > 0:
                state["pbar"].update(delta)
                state["last_done"] = current
            if current >= total:
                state["pbar"].close()
        else:
            # Text progress at 5% increments
            pct = int(current / total * 100) if total else 0
            if (pct % 5 == 0 or pct == 100) and pct != state["last_pct"]:
                print(f"\rProgress: {current}/{total} ({pct}%)", end="", flush=True)
                state["last_pct"] = pct
            if current >= total:
                print()
                state["last_pct"] = -1

    # Run rewrite
    start_time = time.time()
    try:
        success = rewrite_process(
            params=params,
            progress_callback=_progress,
            log_callback=_log_callback if args.verbose else None,
            parallel=args.parallel,
            max_workers=args.max_workers,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

    elapsed = time.time() - start_time

    if success:
        print(f"\nDone in {elapsed:.1f}s. Output: {args.output}")
    else:
        print("\nRewriting completed with errors. Check output for partial results.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
