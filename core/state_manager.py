"""
Atomic state and intermediate file persistence.
"""
import os
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def save_state(filename: str, data: Dict) -> None:
    temp_file = filename + ".tmp"
    try:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, filename)
        logger.debug(f"State saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving state to {filename}: {e}", exc_info=True)
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
        raise


def load_state(filename: str) -> Optional[Dict]:
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            state = json.load(f)
        required = {"processed_block_index", "original_blocks_data", "total_blocks"}
        if not required.issubset(state.keys()):
            logger.warning(f"State file {filename} has incomplete format")
            return None
        return state
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading state from {filename}: {e}", exc_info=True)
        return None


def save_intermediate(filename: str, content: str, context: str = "") -> None:
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_file, filename)
        logger.debug(f"{context}: Intermediate saved: {filename}")
    except Exception as e:
        logger.error(f"{context}: Error saving {filename}: {e}", exc_info=True)
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
        raise
