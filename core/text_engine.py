"""
Text splitting and sentence utilities.
"""
import re
import string
import difflib
from typing import List, Optional, Dict

from core.config import (
    SPLIT_PRIORITY_ENHANCED,
    MIN_BLOCK_LEN_FACTOR,
    MAX_BLOCK_LEN_FACTOR,
    SEARCH_RADIUS_FACTOR,
    MIN_REWRITE_LENGTH_RATIO,
    MAX_REWRITE_LENGTH_RATIO,
    SIMILARITY_THRESHOLD,
    MIN_WORDS_FOR_DUPLICATE_CHECK,
    START_MARKER,
    END_MARKER,
)


class BlockInfo(dict):
    """Block of text to be rewritten."""
    # Fields: block_index, start_char_index, end_char_index,
    #         original_char_length, processed, failed_attempts


def count_chars(text: str) -> int:
    return len(text) if isinstance(text, str) else 0


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def has_more_than_n_words(sentence: str, n: int = 10) -> bool:
    return len(sentence.split()) > n


def normalize_sentence(sentence: str) -> str:
    sentence = sentence.lower().rstrip(string.punctuation)
    return re.sub(r"\s+", " ", sentence).strip()


def calculate_text_quality_metrics(original: str, rewritten: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    orig_len = len(original)
    rew_len = len(rewritten)

    if orig_len == 0 or rew_len == 0:
        metrics["similarity"] = 1.0 if rew_len == 0 else 0.0
        metrics["length_ratio"] = 1.0
        metrics["diversity"] = 0.0
        return metrics

    metrics["length_ratio"] = rew_len / orig_len

    if orig_len > 10000 or rew_len > 10000:
        sample = min(5000, orig_len)
        similarity = difflib.SequenceMatcher(None, original[:sample], rewritten[:sample]).ratio()
    else:
        similarity = difflib.SequenceMatcher(None, original, rewritten).ratio()
    metrics["similarity"] = similarity

    orig_words = set(re.findall(r"\b\w+\b", original.lower()))
    rew_words = set(re.findall(r"\b\w+\b", rewritten.lower()))
    metrics["diversity"] = len(rew_words - orig_words) / max(len(orig_words), 1) if orig_words else 0.0

    return metrics


def validate_rewritten_text(
    text: str,
    original: str,
    orig_len: int,
    prev_block: str,
    next_block: str,
    context: str,
):
    """Returns (is_valid, error_message, quality_metrics)."""
    if not text.strip() and original.strip():
        return False, f"{context}: Empty text for non-empty original.", None

    text_cleaned = text.replace(START_MARKER, "").replace(END_MARKER, "")
    quality_metrics = calculate_text_quality_metrics(original, text_cleaned)

    if original.strip() and text_cleaned.strip() and original != text_cleaned:
        similarity = quality_metrics["similarity"]
        if similarity >= SIMILARITY_THRESHOLD:
            return False, f"{context}: Too similar to original ({similarity:.3f}).", quality_metrics

    if orig_len > 0:
        text_len_cleaned = count_chars(text_cleaned)
        min_len = orig_len * MIN_REWRITE_LENGTH_RATIO
        max_len = orig_len * MAX_REWRITE_LENGTH_RATIO
        if text_len_cleaned > max_len:
            return False, f"{context}: Too long ({text_len_cleaned} > {int(max_len)}).", quality_metrics
        if text_len_cleaned < min_len and orig_len > 20:
            return False, f"{context}: Too short ({text_len_cleaned} < {int(min_len)}).", quality_metrics

    rewritten_sentences = [normalize_sentence(s) for s in split_into_sentences(text_cleaned)]

    if prev_block:
        prev_sentences = {
            normalize_sentence(s)
            for s in split_into_sentences(prev_block)
            if has_more_than_n_words(s, MIN_WORDS_FOR_DUPLICATE_CHECK)
        }
        if any(sent in prev_sentences for sent in rewritten_sentences):
            return False, f"{context}: Repeats sentence(s) from previous block.", quality_metrics

    if next_block:
        next_sentences = {
            normalize_sentence(s)
            for s in split_into_sentences(next_block)
            if has_more_than_n_words(s, MIN_WORDS_FOR_DUPLICATE_CHECK)
        }
        if any(sent in next_sentences for sent in rewritten_sentences):
            return False, f"{context}: Repeats sentence(s) from next block.", quality_metrics

    return True, None, quality_metrics


def find_split_point(text: str, start: int, target_end: int, min_len: int, max_len: int) -> int:
    text_len = len(text)
    ideal_end = min(text_len, max(start + min_len, min(target_end, start + max_len)))

    if ideal_end >= text_len:
        return text_len

    radius = int((target_end - start) * max(SEARCH_RADIUS_FACTOR, 0.15))
    search_start = max(start + min_len, ideal_end - radius)
    search_end = min(text_len, ideal_end + radius, start + max_len)

    best_point = -1
    min_dist = float("inf")
    best_priority = float("inf")

    for priority, seq in enumerate(SPLIT_PRIORITY_ENHANCED):
        pos = search_start
        while pos < search_end:
            try:
                idx = text.index(seq, pos, search_end) + len(seq)
                if idx > start + min_len:
                    dist = abs(idx - ideal_end)
                    score = dist + (priority * 1000)
                    if score < (min_dist + (best_priority * 1000)):
                        min_dist = dist
                        best_point = idx
                        best_priority = priority
                pos = idx + 1
            except ValueError:
                break
        if best_point != -1 and best_priority <= 2:
            return best_point

    if best_point == -1:
        try:
            last_space = text.rindex(" ", search_start, min(ideal_end, start + max_len))
            if last_space > start:
                return last_space + 1
        except ValueError:
            pass

    return best_point if best_point != -1 else min(ideal_end, start + max_len)


def split_into_blocks(text: str, target_size: int) -> Optional[List[BlockInfo]]:
    text_len = count_chars(text)
    if not text_len:
        return None

    blocks = []
    current_pos = 0
    block_idx = 0
    min_len = int(target_size * MIN_BLOCK_LEN_FACTOR)
    max_len = int(target_size * MAX_BLOCK_LEN_FACTOR)

    while current_pos < text_len:
        target_end = current_pos + target_size
        end_pos = find_split_point(text, current_pos, target_end, min_len, max_len)
        if end_pos <= current_pos:
            end_pos = text_len

        blocks.append({
            "block_index": block_idx,
            "start_char_index": current_pos,
            "end_char_index": end_pos,
            "original_char_length": end_pos - current_pos,
            "processed": False,
            "failed_attempts": 0,
        })
        current_pos = end_pos
        block_idx += 1

    return blocks
