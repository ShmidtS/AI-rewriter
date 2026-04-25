"""
Main rewriting orchestration loop.
Decoupled from GUI — works headless or with any frontend.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

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
from core.state_manager import BlockState, RewriteState, StateManager, load_state, save_intermediate
from core.text_engine import count_chars, split_into_blocks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteParams:
    input_file: str
    output_file: str
    language: str
    style: str
    goal: str
    model_name: str
    resume: bool
    save_interval: int
    prompt_preset: str
    base_url: str
    token: str
    output_language: str
    target_language: str


def _normalize_params(params: Mapping[str, Any]) -> RewriteParams:
    input_file = str(params["input_file"])
    output_file = str(params["output_file"])
    language = str(params["language"])
    style = str(params["style"])
    goal = str(params["goal"])
    model_name = str(params["rewriter_model"])
    resume = bool(params.get("resume", True))
    save_interval = int(params.get("save_interval", 1))
    prompt_preset = str(params.get("prompt_preset", "literary"))
    base_url = str(params.get("base_url") or LOCAL_API_BASE_URL)
    token = str(params.get("token") or LOCAL_API_TOKEN)
    output_language = str(params.get("output_language", ""))
    target_language = output_language if output_language and output_language != language else language
    return RewriteParams(
        input_file=input_file,
        output_file=output_file,
        language=language,
        style=style,
        goal=goal,
        model_name=model_name,
        resume=resume,
        save_interval=save_interval,
        prompt_preset=prompt_preset,
        base_url=base_url,
        token=token,
        output_language=output_language,
        target_language=target_language,
    )


def _prepare_output_path(output_path: str, input_path: str) -> str:
    del input_path
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    return output_path


def _get_output_paths(output_path: str) -> tuple[str, str, str, str]:
    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    state_file = os.path.join(output_dir, base_name + STATE_SUFFIX)
    intermediate_file = os.path.join(output_dir, base_name + INTERMEDIATE_SUFFIX)
    return output_dir, base_name, state_file, intermediate_file


def _load_or_init_state(params: RewriteParams, output_path: str) -> tuple[RewriteState | None, int]:
    _, _, state_file, _ = _get_output_paths(output_path)
    if not params.resume:
        return None, -1
    state = load_state(state_file)
    if not state:
        return None, -1
    return state, state.get("processed_block_index", -1)


def _init_blocks(text: str, chunk_size: int, chunk_overlap: int) -> list[BlockState]:
    del chunk_overlap
    blocks = split_into_blocks(text, chunk_size)
    return cast(list[BlockState], blocks or [])


def _rewrite_single_block(
    block: dict[str, Any],
    block_idx: int,
    params: RewriteParams,
    api_fn,
    prompt_fn,
    context: str,
) -> str:
    block_text = block["text"]
    original_block_length = block.get("original_char_length", len(block_text))
    min_len_api = int(original_block_length * MIN_REWRITE_LENGTH_RATIO)
    max_len_api = int(original_block_length * MAX_REWRITE_LENGTH_RATIO)
    system_instr = get_system_prompt(params.prompt_preset, min_len_api, max_len_api)
    user_content = prompt_fn(
        params.target_language,
        params.style,
        params.goal,
        block_text,
        block.get("prev_block_text", ""),
        block.get("next_block_text", ""),
        original_block_length,
        context,
    )
    result = api_fn(
        system_instruction=system_instr,
        user_content=user_content,
        model_name=params.model_name,
        orig_len=original_block_length,
        original=block_text,
        prev_block=block.get("prev_block_text", ""),
        next_block=block.get("next_block_text", ""),
        stop_event=block.get("stop_event"),
        global_context=context,
        failed_attempts=block.get("failed_attempts", 0),
        previous_quality_metrics=block.get("last_quality_metrics"),
        base_url=params.base_url,
        token=params.token,
    )
    if result is None:
        raise RuntimeError(f"Block {block_idx + 1}: API failed")
    new_text, context_update = result
    block["context_update"] = context_update
    return new_text


def _save_progress(state_manager: StateManager, state: RewriteState, block_idx: int, force: bool = False) -> None:
    state["processed_block_index"] = block_idx
    state["timestamp"] = time.time()
    state_manager.set_state(state, processed_chunks=1)
    state_manager.save_if_dirty(force=force)


def _finalize_output(output_path: str, blocks: list[dict[str, str]]) -> str:
    output_dir, base_name, _, _ = _get_output_paths(output_path)
    final_file = os.path.join(output_dir, base_name + FINAL_SUFFIX)
    final_text = "".join(block["text"] for block in blocks)
    save_intermediate(final_file, final_text, "Final")
    return final_file


def rewrite_process(
    params: Mapping[str, Any],
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

    rewrite_params = _normalize_params(params)

    if stop_event is None:
        stop_event = threading.Event()

    output_file = _prepare_output_path(rewrite_params.output_file, rewrite_params.input_file)
    output_dir, base_name, state_file, intermediate_file = _get_output_paths(output_file)
    state_manager = StateManager(state_file, save_interval=rewrite_params.save_interval)

    _log(f"Start: {rewrite_params.input_file} -> {output_file}")
    _log(
        f"Input language: {rewrite_params.language} | Target language: {rewrite_params.target_language} | "
        f"Model: {rewrite_params.model_name} | Preset: {rewrite_params.prompt_preset}"
    )
    if parallel:
        _log(f"Mode: PARALLEL (workers={max_workers if max_workers is not None else 'auto'})")
    else:
        _log("Mode: SEQUENTIAL")

    global_context = GlobalContext()

    # --- Read input ---
    try:
        with open(rewrite_params.input_file, encoding="utf-8") as f:
            original_text = f.read()
        if not original_text.strip():
            _log("Input file is empty.", "error")
            return False
    except FileNotFoundError:
        _log(f"Input file not found: {rewrite_params.input_file}", "error")
        return False
    except OSError as e:
        _log(f"Error reading input: {e}", "error")
        return False

    _log(f"Input length: {count_chars(original_text)} chars")

    blocks: list[BlockState] | None = None
    processed_idx = -1
    rewritten_text: str | None = None
    state_loaded = False

    # --- Resume ---
    state, processed_idx = _load_or_init_state(rewrite_params, output_file)
    if state:
        try:
            with open(intermediate_file) as f:
                rewritten_text = f.read()
            blocks = state["original_blocks_data"]
            state_loaded = True
            _log(f"Resumed from block {processed_idx + 2}")
            if "global_context" in state:
                global_context = GlobalContext.from_json(state["global_context"])
            state_manager.bind_state(state)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            _log(f"Could not load intermediate file: {e}. Starting fresh.", "warning")
            processed_idx = -1

    if not state_loaded:
        rewritten_text = original_text
        blocks = _init_blocks(original_text, BLOCK_TARGET_CHARS, 0)
        if not blocks:
            _log("Failed to split text into blocks.", "error")
            return False
        save_intermediate(intermediate_file, rewritten_text, "Init")
        state_manager.set_state(
            {
                "processed_block_index": -1,
                "original_blocks_data": blocks,
                "total_blocks": len(blocks),
                "timestamp": time.time(),
                "global_context": global_context.to_json(),
            }
        )
        state_manager.save_if_dirty(force=True)

    # blocks is guaranteed non-None at this point (early return if None)
    assert blocks is not None
    assert rewritten_text is not None
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
            state_manager=state_manager,
            intermediate_file=intermediate_file,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
            _log=_log,
            max_workers=max_workers,
            rewrite_params=rewrite_params,
            global_context=global_context,
            processed_idx=processed_idx,
        )

    # --- Sequential mode (original behavior) ---
    for i in range(total_blocks):
        if stop_event.is_set():
            state_manager.save_if_dirty(force=True)
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
            state_manager.set_state(
                {
                    "processed_block_index": processed_idx,
                    "original_blocks_data": blocks,
                    "total_blocks": total_blocks,
                    "timestamp": time.time(),
                    "global_context": global_context.to_json(),
                }
            )
            state_manager.save_if_dirty(force=True)
            break

        _log(f"Block {i + 1}/{total_blocks} [{start}:{end}] ({end - start} chars)")

        block_text = rewritten_text[start:end]
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

        rewrite_block = {
            **block,
            "text": block_text,
            "prev_block_text": prev_block_text,
            "next_block_text": next_block_text,
            "stop_event": stop_event,
        }
        try:
            new_text = _rewrite_single_block(
                rewrite_block,
                i,
                rewrite_params,
                call_local_rewrite_api,
                create_rewrite_prompt,
                global_context,
            )
        except RuntimeError:
            _log(f"Block {i + 1}: API failed. Pausing 10s...", "warning")
            time.sleep(10)
            block["failed_attempts"] += 1
            state_manager.set_state(
                {
                    "processed_block_index": processed_idx,
                    "original_blocks_data": blocks,
                    "total_blocks": total_blocks,
                    "timestamp": time.time(),
                    "global_context": global_context.to_json(),
                }
            )
            state_manager.save_if_dirty(force=True)
            continue

        context_update = rewrite_block["context_update"]

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

        _save_progress(
            state_manager,
            {
                "processed_block_index": processed_idx,
                "original_blocks_data": blocks,
                "total_blocks": total_blocks,
                "global_context": global_context.to_json(),
            },
            processed_idx,
        )

    # Final save
    processed_count = sum(1 for b in blocks if b.get("processed", False))
    failed_count = sum(
        1 for b in blocks if b.get("failed_attempts", 0) >= MAX_RETRIES and not b.get("processed", False)
    )
    _log(f"Done. Processed: {processed_count}/{total_blocks}. Failed: {failed_count}.")

    final_file = _finalize_output(output_file, [{"text": rewritten_text}])
    state_manager.set_state(
        {
            "processed_block_index": processed_idx,
            "original_blocks_data": blocks,
            "total_blocks": total_blocks,
            "timestamp": time.time(),
            "global_context": global_context.to_json(),
        }
    )
    state_manager.save_if_dirty(force=True)
    _log(f"Final result: {final_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)

    return True


def _rewrite_parallel(
    *,
    blocks: list[BlockState],
    total_blocks: int,
    original_text: str,
    output_dir: str,
    base_name: str,
    state_manager: StateManager,
    intermediate_file: str,
    stop_event: threading.Event,
    progress_callback: Callable[[int, int], None] | None,
    log_callback: Callable[[str], None] | None,
    _log,
    max_workers: int | None,
    rewrite_params: RewriteParams,
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

        # Thread-safe: copy global_context under lock to avoid race during read
        with _context_lock:
            context_snapshot = global_context.to_json()
        rewrite_block = {
            **block,
            "text": block_text,
            "prev_block_text": prev_block_text,
            "next_block_text": next_block_text,
            "stop_event": stop_event,
        }
        try:
            new_text = _rewrite_single_block(
                rewrite_block,
                block_index,
                rewrite_params,
                call_local_rewrite_api,
                create_rewrite_prompt,
                GlobalContext.from_json(context_snapshot),
            )
        except RuntimeError:
            _log(f"Block {block_index + 1}: API failed.", "warning")
            block["failed_attempts"] += 1
            with results_lock:
                progress_counter[0] += 1
            _report_progress(progress_counter[0], len(already_done))
            return False

        context_update = rewrite_block["context_update"]

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

    final_file = _finalize_output(os.path.join(output_dir, base_name), [{"text": final_text}])
    final_processed_idx = total_blocks - 1 if processed_count == total_blocks else processed_idx
    state_manager.set_state(
        {
            "processed_block_index": final_processed_idx,
            "original_blocks_data": blocks,
            "total_blocks": total_blocks,
            "timestamp": time.time(),
            "global_context": global_context.to_json(),
        }
    )
    state_manager.save_if_dirty(force=True)
    _log(f"Final result: {final_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)

    return True
