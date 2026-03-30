"""
Main rewriting orchestration loop.
Decoupled from GUI — works headless or with any frontend.
"""
import os
import time
import logging
import threading
from typing import Dict, Optional, Callable

from core.config import (
    BLOCK_TARGET_CHARS,
    MAX_RETRIES,
    STATE_SUFFIX,
    INTERMEDIATE_SUFFIX,
    FINAL_SUFFIX,
    START_MARKER,
    END_MARKER,
    LOCAL_API_BASE_URL,
    LOCAL_API_TOKEN,
    MIN_REWRITE_LENGTH_RATIO,
    MAX_REWRITE_LENGTH_RATIO,
)
from core.text_engine import split_into_blocks, count_chars
from core.context import GlobalContext
from core.prompts import get_system_prompt, create_rewrite_prompt
from core.api_client import call_local_rewrite_api
from core.state_manager import save_state, load_state, save_intermediate

logger = logging.getLogger(__name__)


def rewrite_process(
    params: Dict,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_event: Optional[threading.Event] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Main rewriting loop.

    params keys:
        input_file, output_file, language, style, goal,
        rewriter_model, resume, save_interval,
        prompt_preset (optional, default 'literary'),
        base_url (optional), token (optional)

    Returns True on completion, False on fatal error.
    """
    def _log(msg: str, level: str = "info"):
        getattr(logger, level)(msg)
        if log_callback:
            log_callback(msg)

    input_file  = params["input_file"]
    output_file = params["output_file"]
    language    = params["language"]
    style       = params["style"]
    goal        = params["goal"]
    model_name  = params["rewriter_model"]
    resume      = params.get("resume", True)
    save_interval = params.get("save_interval", 1)
    prompt_preset = params.get("prompt_preset", "literary")
    base_url    = params.get("base_url", LOCAL_API_BASE_URL)
    token       = params.get("token", LOCAL_API_TOKEN)

    if stop_event is None:
        stop_event = threading.Event()

    output_dir = os.path.dirname(output_file) or "."
    base_name  = os.path.splitext(os.path.basename(output_file))[0]
    state_file        = os.path.join(output_dir, base_name + STATE_SUFFIX)
    intermediate_file = os.path.join(output_dir, base_name + INTERMEDIATE_SUFFIX)

    _log(f"Start: {input_file} -> {output_file}")
    _log(f"Language: {language} | Model: {model_name} | Preset: {prompt_preset}")

    global_context = GlobalContext()

    # --- Read input ---
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            original_text = f.read()
        if not original_text.strip():
            _log("Input file is empty.", "error")
            return False
    except FileNotFoundError:
        _log(f"Input file not found: {input_file}", "error")
        return False
    except Exception as e:
        _log(f"Error reading input: {e}", "error")
        return False

    _log(f"Input length: {count_chars(original_text)} chars")

    blocks = None
    processed_idx = -1
    rewritten_text = None
    state_loaded = False

    # --- Resume ---
    if resume:
        state = load_state(state_file)
        if state:
            try:
                with open(intermediate_file, "r", encoding="utf-8") as f:
                    rewritten_text = f.read()
                blocks = state["original_blocks_data"]
                processed_idx = state.get("processed_block_index", -1)
                state_loaded = True
                _log(f"Resumed from block {processed_idx + 2}")
                if "global_context" in state:
                    global_context = GlobalContext.from_json(state["global_context"])
            except Exception as e:
                _log(f"Could not load intermediate file: {e}. Starting fresh.", "warning")

    if not state_loaded:
        rewritten_text = original_text
        blocks = split_into_blocks(original_text, BLOCK_TARGET_CHARS)
        if not blocks:
            _log("Failed to split text into blocks.", "error")
            return False
        save_intermediate(intermediate_file, rewritten_text, "Init")
        save_state(state_file, {
            "processed_block_index": -1,
            "original_blocks_data": blocks,
            "total_blocks": len(blocks),
            "timestamp": time.time(),
            "global_context": global_context.to_json(),
        })

    total_blocks = len(blocks)
    _log(f"Total blocks: {total_blocks}")
    if progress_callback:
        progress_callback(processed_idx + 1, total_blocks)

    for i in range(total_blocks):
        if stop_event.is_set():
            _log("Stopped by user.")
            break

        block = blocks[i]
        if i <= processed_idx or block.get("processed", False):
            continue
        if block.get("failed_attempts", 0) >= MAX_RETRIES:
            _log(f"Block {i+1}: skipped (max retries exceeded)", "warning")
            continue

        start = block["start_char_index"]
        end   = block["end_char_index"]
        cur_len = count_chars(rewritten_text)

        if not (0 <= start <= end <= cur_len):
            _log(f"Block {i+1}: invalid bounds [{start}:{end}] (text len {cur_len})", "error")
            save_state(state_file, {
                "processed_block_index": processed_idx,
                "original_blocks_data": blocks,
                "total_blocks": total_blocks,
                "timestamp": time.time(),
                "global_context": global_context.to_json(),
            })
            break

        _log(f"Block {i+1}/{total_blocks} [{start}:{end}] ({end-start} chars)")

        block_text = rewritten_text[start:end]
        original_block_length = block.get("original_char_length", len(block_text))

        # Context windows
        prev_block_text = ""
        if i > 0:
            pb = blocks[i - 1]
            ps, pe = pb["start_char_index"], pb["end_char_index"]
            if 0 <= ps <= pe <= cur_len:
                prev_block_text = rewritten_text[ps:pe]

        next_block_text = ""
        if i < total_blocks - 1:
            nb = blocks[i + 1]
            ns, ne = nb["start_char_index"], min(nb["end_char_index"], cur_len)
            if 0 <= ns <= ne <= cur_len:
                next_block_text = rewritten_text[ns:ne]

        min_len_api = int(original_block_length * MIN_REWRITE_LENGTH_RATIO)
        max_len_api = int(original_block_length * MAX_REWRITE_LENGTH_RATIO)

        system_instr = get_system_prompt(prompt_preset, min_len_api, max_len_api)
        user_content = create_rewrite_prompt(
            language, style, goal,
            block_text, prev_block_text, next_block_text,
            original_block_length, global_context,
        )

        result = call_local_rewrite_api(
            system_instruction=system_instr,
            user_content=user_content,
            model_name=model_name,
            orig_len=original_block_length,
            original=block_text,
            prev_block=prev_block_text,
            next_block=next_block_text,
            stop_event=stop_event,
            global_context=global_context,
            failed_attempts=block.get("failed_attempts", 0),
            previous_quality_metrics=block.get("last_quality_metrics"),
            base_url=base_url,
            token=token,
        )

        if result is None:
            _log(f"Block {i+1}: API failed. Pausing 10s...", "warning")
            time.sleep(10)
            block["failed_attempts"] += 1
            continue

        new_text, context_update = result

        if context_update:
            global_context.update_from_response(context_update)

        new_text_len = count_chars(new_text)
        delta = new_text_len - len(block_text)
        _log(f"Block {i+1} done. Delta: {delta:+d} chars")

        rewritten_text = rewritten_text[:start] + new_text + rewritten_text[end:]
        block["end_char_index"] = start + new_text_len
        block["processed"] = True
        block["failed_attempts"] = 0
        block["last_quality_metrics"] = None
        processed_idx = i

        save_intermediate(intermediate_file, rewritten_text, f"Block {i+1}")

        if delta != 0:
            for j in range(i + 1, total_blocks):
                blocks[j]["start_char_index"] += delta
                blocks[j]["end_char_index"]   += delta

        if progress_callback:
            progress_callback(i + 1, total_blocks)

        if save_interval and (i + 1) % save_interval == 0:
            save_state(state_file, {
                "processed_block_index": processed_idx,
                "original_blocks_data": blocks,
                "total_blocks": total_blocks,
                "timestamp": time.time(),
                "global_context": global_context.to_json(),
            })

    # Final save
    processed_count = sum(1 for b in blocks if b.get("processed", False))
    failed_count    = sum(1 for b in blocks if b.get("failed_attempts", 0) >= MAX_RETRIES and not b.get("processed", False))
    _log(f"Done. Processed: {processed_count}/{total_blocks}. Failed: {failed_count}.")

    final_file = os.path.join(output_dir, base_name + FINAL_SUFFIX)
    save_intermediate(final_file, rewritten_text, "Final")
    save_state(state_file, {
        "processed_block_index": processed_idx,
        "original_blocks_data": blocks,
        "total_blocks": total_blocks,
        "timestamp": time.time(),
        "global_context": global_context.to_json(),
    })
    _log(f"Final result: {final_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)

    return True
