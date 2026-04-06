"""
Main rewriting orchestration loop.
Decoupled from GUI — works headless or with any frontend.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from core.api_client import call_local_rewrite_api
from core.config import (
    BLOCK_TARGET_CHARS,
    FINAL_SUFFIX,
    INTERMEDIATE_SUFFIX,
    LOCAL_API_BASE_URL,
    LOCAL_API_TOKEN,
    MAX_RETRIES,
    MAX_REWRITE_LENGTH_RATIO,
    MIN_REWRITE_LENGTH_RATIO,
    STATE_SUFFIX,
)
from core.context import GlobalContext
from core.prompts import create_rewrite_prompt, get_system_prompt
from core.state_manager import load_state, save_intermediate, save_state
from core.text_engine import count_chars, split_into_blocks

logger = logging.getLogger(__name__)


def rewrite_process(
    params: dict,
    progress_callback: Callable[[int, int], None] | None = None,
    stop_event: threading.Event | None = None,
    log_callback: Callable[[str], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = 10,
) -> bool:
    """
    Main rewriting loop.

    params keys:
        input_file, output_file, language, style, goal,
        rewriter_model, resume, save_interval,
        prompt_preset (optional, default 'literary'),
        base_url (optional), token (optional)

    parallel:
        When False (default), processes blocks sequentially — unchanged behavior.
        When True, processes blocks concurrently using ThreadPoolExecutor.
        Each block is rewritten independently; no context sharing between threads.
        Results are collected and concatenated in order after all blocks complete.

    Returns True on completion, False on fatal error.
    """

    def _log(msg: str, level: str = "info"):
        getattr(logger, level)(msg)
        if log_callback:
            log_callback(msg)

    input_file = params["input_file"]
    output_file = params["output_file"]
    language = params["language"]
    style = params["style"]
    goal = params["goal"]
    model_name = params["rewriter_model"]
    resume = params.get("resume", True)
    save_interval = params.get("save_interval", 1)
    prompt_preset = params.get("prompt_preset", "literary")
    base_url = params.get("base_url", LOCAL_API_BASE_URL)
    token = params.get("token", LOCAL_API_TOKEN)
    output_language = params.get("output_language", "")
    # When output_language is explicitly set and differs from input language,
    # use output_language as the target language for the rewrite prompt
    target_language = output_language if output_language and output_language != language else language

    if stop_event is None:
        stop_event = threading.Event()

    output_dir = os.path.dirname(output_file) or "."
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    state_file = os.path.join(output_dir, base_name + STATE_SUFFIX)
    intermediate_file = os.path.join(output_dir, base_name + INTERMEDIATE_SUFFIX)

    _log(f"Start: {input_file} -> {output_file}")
    _log(f"Input language: {language} | Target language: {target_language} | Model: {model_name} | Preset: {prompt_preset}")
    if parallel:
        _log(f"Mode: PARALLEL (workers={max_workers if max_workers is not None else 'auto'})")
    else:
        _log("Mode: SEQUENTIAL")

    global_context = GlobalContext()

    # --- Read input ---
    try:
        with open(input_file, encoding="utf-8") as f:
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

    blocks: list[dict] | None = None
    processed_idx = -1
    rewritten_text: str | None = None
    state_loaded = False

    # --- Resume ---
    if resume:
        state = load_state(state_file)
        if state:
            try:
                with open(intermediate_file) as f:
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
        save_state(
            state_file,
            {
                "processed_block_index": -1,
                "original_blocks_data": blocks,
                "total_blocks": len(blocks),
                "timestamp": time.time(),
                "global_context": global_context.to_json(),
            },
        )

    # blocks is guaranteed non-None at this point (early return if None)
    total_blocks = len(blocks)
    _log(f"Total blocks: {total_blocks}")
    if progress_callback:
        progress_callback(processed_idx + 1, total_blocks)

    # --- Parallel mode ---
    if parallel:
        return _rewrite_parallel(
            blocks=blocks,
            total_blocks=total_blocks,
            original_text=original_text,
            output_dir=output_dir,
            base_name=base_name,
            state_file=state_file,
            intermediate_file=intermediate_file,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
            _log=_log,
            max_workers=max_workers,
            language=target_language,
            style=style,
            goal=goal,
            model_name=model_name,
            prompt_preset=prompt_preset,
            base_url=base_url,
            token=token,
            save_interval=save_interval,
            global_context=global_context,
            processed_idx=processed_idx,
        )

    # --- Sequential mode (original behavior) ---
    for i in range(total_blocks):
        if stop_event.is_set():
            _log("Stopped by user.")
            break

        block = blocks[i]
        if i <= processed_idx or block.get("processed", False):
            continue
        if block.get("failed_attempts", 0) >= MAX_RETRIES:
            _log(f"Block {i + 1}: skipped (max retries exceeded)", "warning")
            continue

        start = block["start_char_index"]
        end = block["end_char_index"]
        cur_len = count_chars(rewritten_text)

        if not (0 <= start <= end <= cur_len):
            _log(
                f"Block {i + 1}: invalid bounds [{start}:{end}] (text len {cur_len})",
                "error",
            )
            save_state(
                state_file,
                {
                    "processed_block_index": processed_idx,
                    "original_blocks_data": blocks,
                    "total_blocks": total_blocks,
                    "timestamp": time.time(),
                    "global_context": global_context.to_json(),
                },
            )
            break

        _log(f"Block {i + 1}/{total_blocks} [{start}:{end}] ({end - start} chars)")

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
            target_language,
            style,
            goal,
            block_text,
            prev_block_text,
            next_block_text,
            original_block_length,
            global_context,
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
            _log(f"Block {i + 1}: API failed. Pausing 10s...", "warning")
            time.sleep(10)
            block["failed_attempts"] += 1
            continue

        new_text, context_update = result

        if context_update:
            global_context.update_from_response(context_update)

        new_text_len = count_chars(new_text)
        delta = new_text_len - len(block_text)
        _log(f"Block {i + 1} done. Delta: {delta:+d} chars")

        rewritten_text = rewritten_text[:start] + new_text + rewritten_text[end:]
        block["end_char_index"] = start + new_text_len
        block["processed"] = True
        block["failed_attempts"] = 0
        block["last_quality_metrics"] = None
        processed_idx = i

        save_intermediate(intermediate_file, rewritten_text, f"Block {i + 1}")

        if delta != 0:
            for j in range(i + 1, total_blocks):
                blocks[j]["start_char_index"] += delta
                blocks[j]["end_char_index"] += delta

        if progress_callback:
            progress_callback(i + 1, total_blocks)

        if save_interval and (i + 1) % save_interval == 0:
            save_state(
                state_file,
                {
                    "processed_block_index": processed_idx,
                    "original_blocks_data": blocks,
                    "total_blocks": total_blocks,
                    "timestamp": time.time(),
                    "global_context": global_context.to_json(),
                },
            )

    # Final save
    processed_count = sum(1 for b in blocks if b.get("processed", False))
    failed_count = sum(
        1 for b in blocks if b.get("failed_attempts", 0) >= MAX_RETRIES and not b.get("processed", False)
    )
    _log(f"Done. Processed: {processed_count}/{total_blocks}. Failed: {failed_count}.")

    final_file = os.path.join(output_dir, base_name + FINAL_SUFFIX)
    save_intermediate(final_file, rewritten_text, "Final")
    save_state(
        state_file,
        {
            "processed_block_index": processed_idx,
            "original_blocks_data": blocks,
            "total_blocks": total_blocks,
            "timestamp": time.time(),
            "global_context": global_context.to_json(),
        },
    )
    _log(f"Final result: {final_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)

    return True


def _rewrite_parallel(
    *,
    blocks: list,
    total_blocks: int,
    original_text: str,
    output_dir: str,
    base_name: str,
    state_file: str,
    intermediate_file: str,
    stop_event: threading.Event,
    progress_callback: Callable[[int, int], None] | None,
    log_callback: Callable[[str], None] | None,
    _log,
    max_workers: int | None,
    language: str,
    style: str,
    goal: str,
    model_name: str,
    prompt_preset: str,
    base_url: str,
    token: str,
    save_interval: int,
    global_context: GlobalContext,
    processed_idx: int,
) -> bool:
    """
    Parallel rewrite implementation using ThreadPoolExecutor.

    Strategy: each thread rewrites a single block independently.
    Since parallel execution changes text lengths unpredictably,
    results are collected in order and concatenated after all workers finish.

    No context sharing between concurrent blocks — each block gets its own
    context window from the original text.
    """
    import concurrent.futures

    # Determine worker count
    effective_workers = max_workers if max_workers is not None else min(4, total_blocks)

    _log(f"PARALLEL mode: {total_blocks} blocks, max_workers={effective_workers}")

    # Results storage: block_index -> rewritten_text, protected by a lock
    results_lock = threading.Lock()
    _context_lock = threading.Lock()
    results: dict[int, str] = {}

    # Track which blocks are already done (from previous sessions)
    already_done = {
        i
        for i in range(total_blocks)
        if i <= processed_idx or blocks[i].get("processed", False) or blocks[i].get("failed_attempts", 0) >= MAX_RETRIES
    }
    blocks_to_process = sorted(i for i in range(total_blocks) if i not in already_done)

    # Use a mutable list for progress counter (need mutable for closure)
    progress_counter = [0]

    def _rewrite_one_block(block_index: int) -> bool:
        """Worker function: rewrite a single block. Returns True on success."""
        if stop_event.is_set():
            return False

        block = blocks[block_index]
        start = block["start_char_index"]
        end = block["end_char_index"]
        original_block_length = block.get("original_char_length", end - start)

        _log(f"Block {block_index + 1}/{total_blocks} [{start}:{end}] ({end - start} chars)")

        # In parallel mode, extract block text from the ORIGINAL text
        block_text = original_text[start:end]

        # Context windows from original text (no shared state between threads)
        prev_block_text = ""
        if block_index > 0:
            pb = blocks[block_index - 1]
            ps, pe = pb["start_char_index"], pb["end_char_index"]
            if 0 <= ps <= pe <= len(original_text):
                prev_block_text = original_text[ps:pe]

        next_block_text = ""
        if block_index < total_blocks - 1:
            nb = blocks[block_index + 1]
            ns, ne = nb["start_char_index"], min(nb["end_char_index"], len(original_text))
            if 0 <= ns <= ne <= len(original_text):
                next_block_text = original_text[ns:ne]

        min_len_api = int(original_block_length * MIN_REWRITE_LENGTH_RATIO)
        max_len_api = int(original_block_length * MAX_REWRITE_LENGTH_RATIO)

        system_instr = get_system_prompt(prompt_preset, min_len_api, max_len_api)
        # Thread-safe: copy global_context under lock to avoid race during read
        with _context_lock:
            context_snapshot = global_context.to_json()
        user_content = create_rewrite_prompt(
            language,
            style,
            goal,
            block_text,
            prev_block_text,
            next_block_text,
            original_block_length,
            GlobalContext.from_json(context_snapshot),
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
            _log(f"Block {block_index + 1}: API failed.", "warning")
            block["failed_attempts"] += 1
            with results_lock:
                progress_counter[0] += 1
            _report_progress(progress_counter[0], len(already_done))
            return False

        new_text, context_update = result

        if context_update:
            global_context.update_from_response(context_update)

        new_text_len = count_chars(new_text)
        _log(f"Block {block_index + 1} done. New len: {new_text_len} chars")

        # Store result with lock
        with results_lock:
            results[block_index] = new_text
            progress_counter[0] += 1

        # Report progress
        _report_progress(progress_counter[0], len(already_done))

        # Mark block as processed
        block["processed"] = True
        block["failed_attempts"] = 0
        block["last_quality_metrics"] = None

        return True

    def _report_progress(completed_this_run: int, already: int) -> None:
        """Thread-safe progress reporting."""
        if progress_callback:
            total_completed = already + completed_this_run
            progress_callback(total_completed, total_blocks)

    # --- Dispatch workers ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for block_index in blocks_to_process:
            future = executor.submit(_rewrite_one_block, block_index)
            futures[future] = block_index

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # --- Check for stop during execution ---
    if stop_event.is_set():
        _log("Stopped by user during parallel rewrite.")

    # --- Assemble final text in block order ---
    block_segments = []
    for i in range(total_blocks):
        if i in already_done:
            # Already-processed: use text from original text at these indices
            b = blocks[i]
            block_segments.append(original_text[b["start_char_index"] : b["end_char_index"]])
        elif i in results:
            # Successfully rewritten
            block_segments.append(results[i])
        else:
            # Not rewritten (failed/stopped) -- fallback to original
            b = blocks[i]
            block_segments.append(original_text[b["start_char_index"] : b["end_char_index"]])
            _log(f"Block {i + 1}: not rewritten, using original", "warning")

    final_text = "".join(block_segments)
    _log(f"Assembled final text: {count_chars(final_text)} chars")

    # --- Save intermediate and final ---
    save_intermediate(intermediate_file, final_text, "Parallel rewrite")

    processed_count = sum(1 for b in blocks if b.get("processed", False))
    failed_count = sum(
        1 for b in blocks if b.get("failed_attempts", 0) >= MAX_RETRIES and not b.get("processed", False)
    )
    _log(f"Done. Processed: {processed_count}/{total_blocks}. Failed: {failed_count}.")

    final_file = os.path.join(output_dir, base_name + FINAL_SUFFIX)
    save_intermediate(final_file, final_text, "Final")
    final_processed_idx = total_blocks - 1 if processed_count == total_blocks else processed_idx
    save_state(
        state_file,
        {
            "processed_block_index": final_processed_idx,
            "original_blocks_data": blocks,
            "total_blocks": total_blocks,
            "timestamp": time.time(),
            "global_context": global_context.to_json(),
        },
    )
    _log(f"Final result: {final_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)

    return True
