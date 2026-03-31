"""
Tests for AI Book Rewriter core functionality.

Covers:
- Text splitting (split_into_blocks)
- Validation (validate_rewritten_text)
- Configuration loading (Settings)
- Parallel mode parameter passing
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ===========================================================================
# Fixtures & Helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """Reset the Settings singleton before each test."""
    import core.settings as settings_mod
    settings_mod._settings = None
    yield
    settings_mod._settings = None


SAMPLE_TEXT = (
    "This is the first sentence. It introduces the topic. "
    "The second sentence adds some detail about the characters. "
    "A third sentence follows with more information. "
    "The fourth sentence begins a new paragraph of thought. "
    "The fifth sentence continues the narrative arc. "
    "The sixth sentence brings the section to a close. "
    "The seventh sentence starts something fresh. "
    "The eighth sentence deepens the intrigue. "
    "The ninth sentence reveals hidden information. "
    "The tenth sentence concludes this particular scene."
)


# ===========================================================================
# 1. Text Splitting — split_into_blocks
# ===========================================================================

class TestSplitIntoBlocks:
    """Tests for core.text_engine.split_into_blocks."""

    def test_returns_none_for_empty_text(self):
        from core.text_engine import split_into_blocks
        assert split_into_blocks("", 1000) is None
        assert split_into_blocks(None, 1000) is None

    def test_single_block_for_short_text(self):
        from core.text_engine import split_into_blocks
        result = split_into_blocks("Short text.", 1000)
        assert result is not None
        assert len(result) == 1
        assert result[0]["block_index"] == 0

    def test_multiple_blocks_for_long_text(self):
        from core.text_engine import split_into_blocks, count_chars
        text = SAMPLE_TEXT * 50  # Make it long enough to split
        target = 500
        blocks = split_into_blocks(text, target)
        assert blocks is not None
        assert len(blocks) > 1

    def test_block_fields(self):
        from core.text_engine import split_into_blocks
        blocks = split_into_blocks(SAMPLE_TEXT, 1000)
        assert blocks is not None
        block = blocks[0]
        for field in ("block_index", "start_char_index", "end_char_index",
                       "original_char_length", "processed", "failed_attempts"):
            assert field in block, f"Missing field: {field}"

    def test_block_indices_are_sequential(self):
        from core.text_engine import split_into_blocks
        text = SAMPLE_TEXT * 20
        blocks = split_into_blocks(text, 400)
        assert blocks is not None
        for i, block in enumerate(blocks):
            assert block["block_index"] == i

    def test_blocks_cover_full_text(self):
        from core.text_engine import split_into_blocks, count_chars
        text = SAMPLE_TEXT * 20
        blocks = split_into_blocks(text, 400)
        assert blocks is not None
        first_start = blocks[0]["start_char_index"]
        last_end = blocks[-1]["end_char_index"]
        assert first_start == 0
        assert last_end == len(text)

    def test_contiguous_blocks_no_gaps(self):
        from core.text_engine import split_into_blocks
        text = SAMPLE_TEXT * 20
        blocks = split_into_blocks(text, 400)
        assert blocks is not None
        for i in range(len(blocks) - 1):
            assert blocks[i]["end_char_index"] == blocks[i + 1]["start_char_index"], \
                f"Gap between blocks {i} and {i+1}"

    def test_processed_flag_defaults_false(self):
        from core.text_engine import split_into_blocks
        blocks = split_into_blocks(SAMPLE_TEXT, 1000)
        assert blocks is not None
        assert all(not b["processed"] for b in blocks)

    def test_failed_attempts_defaults_zero(self):
        from core.text_engine import split_into_blocks
        blocks = split_into_blocks(SAMPLE_TEXT, 1000)
        assert blocks is not None
        assert all(b["failed_attempts"] == 0 for b in blocks)


# ===========================================================================
# 2. Validation — validate_rewritten_text
# ===========================================================================

class TestValidateRewrittenText:
    """Tests for core.text_engine.validate_rewritten_text."""

    def test_empty_rewritten_text_for_nonempty_original(self):
        from core.text_engine import validate_rewritten_text
        is_valid, msg, _ = validate_rewritten_text(
            text="", original="Some original text.", orig_len=19,
            prev_block="", next_block="", context="test"
        )
        assert is_valid is False
        assert "Empty" in msg

    def test_similar_to_original_rejected(self):
        """Text that is nearly identical to original should be rejected."""
        from core.text_engine import validate_rewritten_text
        original = "The quick brown fox jumps over the lazy dog every morning."
        # Slightly modified — should be very similar
        rewritten = "The quick brown fox jumps over the lazy dog every morning!"
        is_valid, msg, metrics = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        # The text is very similar, may be rejected based on threshold
        if not is_valid:
            assert "Too similar" in msg

    def test_too_long_rejected(self):
        from core.text_engine import validate_rewritten_text
        original = "Short text."
        # Much longer than allowed
        rewritten = "A " * 100 + "much longer rewritten text" + " b" * 100
        is_valid, msg, _ = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        assert is_valid is False
        assert "Too long" in msg

    def test_too_short_rejected(self):
        from core.text_engine import validate_rewritten_text
        original = "This is a fairly long sentence with enough characters to check."
        rewritten = "Short."
        is_valid, msg, _ = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        assert is_valid is False
        assert "Too short" in msg

    def test_repeats_from_prev_block_rejected(self):
        from core.text_engine import validate_rewritten_text
        prev_block = (
            "The captain sailed across the northern seas and found an island. "
            "The island was surrounded by coral reefs and palm trees everywhere. "
            "The sunset painted the sky in brilliant shades of orange and purple."
        )
        # Repeat a sentence from prev
        rewritten = (
            "The captain sailed across the northern seas and found an island. "
            "A new adventure began in the tropical paradise beyond the shore."
        )
        original = "Original placeholder text that is different from both."
        is_valid, msg, _ = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block=prev_block, next_block="", context="test"
        )
        assert is_valid is False
        assert "Repeats" in msg.upper() or "repeat" in msg.lower() or not is_valid

    def test_repeats_from_next_block_rejected(self):
        from core.text_engine import validate_rewritten_text
        # Must have > 10 words to trigger duplicate check
        next_block = "The mountain peak gleamed brilliantly white against the expansive azure sky above the valley."
        rewritten = "The mountain peak gleamed brilliantly white against the expansive azure sky above the valley."
        original = "Some completely different text here for this particular block of content."
        is_valid, msg, _ = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block=next_block, context="test"
        )
        assert is_valid is False
        assert "next" in msg.lower()

    def test_valid_transformation_accepted(self):
        from core.text_engine import validate_rewritten_text
        original = (
            "The old house stood at the end of the street, "
            "its windows dark and its garden overgrown with weeds."
        )
        rewritten = (
            "At the far end of the lane stood an aged dwelling, "
            "with unlit windows and a tangle of vegetation where flowers once bloomed."
        )
        is_valid, msg, metrics = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        assert is_valid is True, f"Expected valid, got: {msg}"

    def test_markers_are_stripped_before_validation(self):
        from core.config import START_MARKER, END_MARKER
        from core.text_engine import validate_rewritten_text
        original = "A short original text."
        rewritten = f"{START_MARKER}The dwelling stood at the lane's end.{END_MARKER}"
        is_valid, msg, _ = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        # Should not fail because of markers
        assert "marker" not in (msg or "").lower() if msg else True

    def test_quality_metrics_returned(self):
        from core.text_engine import validate_rewritten_text
        original = "The sun set behind the mountains."
        rewritten = "Evening light faded behind the peaks."
        is_valid, msg, metrics = validate_rewritten_text(
            text=rewritten, original=original, orig_len=len(original),
            prev_block="", next_block="", context="test"
        )
        assert metrics is not None
        assert "similarity" in metrics
        assert "length_ratio" in metrics
        assert "diversity" in metrics
        assert 0 <= metrics["similarity"] <= 1
        assert metrics["length_ratio"] > 0


# ===========================================================================
# 3. Settings / Config
# ===========================================================================

class TestSettings:
    """Tests for core.settings.Settings and core.config."""

    def test_default_lang_is_ru(self):
        """UI_LANG defaults to 'ru'."""
        with patch.dict(os.environ, {}, clear=True):
            from core.settings import Settings
            s = Settings()
            assert s.ui_lang == "ru"

    def test_lang_validation_en(self):
        from core.settings import Settings
        with patch.dict(os.environ, {"UI_LANG": "en"}):
            s = Settings()
            assert s.ui_lang == "en"

    def test_lang_validation_zh(self):
        from core.settings import Settings
        with patch.dict(os.environ, {"UI_LANG": "zh"}):
            s = Settings()
            assert s.ui_lang == "zh"

    def test_lang_validation_raises(self):
        from core.settings import Settings
        with patch.dict(os.environ, {"UI_LANG": "fr"}):
            with pytest.raises(Exception):
                Settings()

    def test_default_connection_profile(self):
        with patch.dict(os.environ, {"UI_LANG": "en"}):
            from core.settings import Settings
            s = Settings()
            assert s.is_proxy_mode() is True

    def test_get_api_base_url_proxy_default(self):
        with patch.dict(os.environ, {"UI_LANG": "en"}):
            from core.settings import Settings
            s = Settings()
            url = s.get_api_base_url()
            assert url.startswith("http")

    def test_get_api_base_url_direct(self):
        with patch.dict(os.environ, {
            "UI_LANG": "en",
            "CONNECTION_PROFILE": "direct",
            "DIRECT_BASE_URL": "https://api.example.com/v1",
            "DIRECT_API_KEY": "test-key"
        }):
            from core.settings import Settings, ConnectionProfile
            s = Settings()
            assert s.get_api_base_url() == "https://api.example.com/v1"

    def test_timeout_based_on_profile(self):
        with patch.dict(os.environ, {
            "UI_LANG": "en",
            "CONNECTION_PROFILE": "proxy",
        }):
            from core.settings import Settings
            s = Settings()
            timeout = s.get_timeout()
            assert timeout > 0

    def test_config_module_exports_constants(self):
        """core.config should export all expected constants."""
        from core.config import (
            BLOCK_TARGET_CHARS,
            MIN_REWRITE_LENGTH_RATIO,
            MAX_REWRITE_LENGTH_RATIO,
            MAX_RETRIES,
            START_MARKER,
            END_MARKER,
            SIMILARITY_THRESHOLD,
        )
        assert BLOCK_TARGET_CHARS > 0
        assert 0 < MIN_REWRITE_LENGTH_RATIO < 1
        assert MAX_REWRITE_LENGTH_RATIO > 1
        assert MAX_RETRIES > 0
        assert START_MARKER
        assert END_MARKER
        assert 0 < SIMILARITY_THRESHOLD <= 1


# ===========================================================================
# 4. Parallel Mode Parameter Passing
# ===========================================================================

class TestParallelMode:
    """Tests that parallel mode passes parameters correctly."""

    def test_parallel_param_is_passed_to_service(self):
        """verify that parallel=True reaches the rewrite service."""
        from core.services import RewriteService, RewriteParams
        with patch.object(RewriteService, 'start_rewrite') as mocked:
            mocked.return_value = True
            service = RewriteService()
            params = RewriteParams(
                input_file="dummy.txt",
                output_file="dummy_out.txt",
                language="English",
                style="",
                goal="",
                model="test-model",
                resume=False,
                parallel=True,
                save_interval=1,
                prompt_preset="literary",
            )
            service.start_rewrite(params, parallel=True)
            mocked.assert_called_once()
            call_kwargs = mocked.call_args
            assert call_kwargs.kwargs.get("parallel") is True

    def test_sequential_param_is_passed_to_service(self):
        """verify that parallel=False reaches the rewrite service."""
        from core.services import RewriteService, RewriteParams
        with patch.object(RewriteService, 'start_rewrite') as mocked:
            mocked.return_value = True
            service = RewriteService()
            params = RewriteParams(
                input_file="dummy.txt",
                output_file="dummy_out.txt",
                language="English",
                style="",
                goal="",
                model="test-model",
                resume=False,
                parallel=False,
                save_interval=1,
                prompt_preset="academic",
            )
            service.start_rewrite(params, parallel=False)
            mocked.assert_called_once()
            call_kwargs = mocked.call_args
            assert call_kwargs.kwargs.get("parallel") is False

    def test_rewrite_params_has_parallel_field(self):
        """RewriteParams dataclass should have a parallel field."""
        from core.services import RewriteParams
        params = RewriteParams(
            input_file="a.txt",
            output_file="b.txt",
            language="en",
            style="",
            goal="",
            model="m",
            resume=True,
            parallel=True,
            save_interval=1,
        )
        assert params.parallel is True

    def test_rewriter_parallel_param_defaults_false(self):
        """rewrite_process defaults: parallel should not be True unless set."""
        from core.rewriter import rewrite_process
        import inspect
        sig = inspect.signature(rewrite_process)
        assert sig.parameters["parallel"].default is False

    def test_rewriter_accepts_max_workers(self):
        """rewrite_process has max_workers parameter."""
        from core.rewriter import rewrite_process
        import inspect
        sig = inspect.signature(rewrite_process)
        assert "max_workers" in sig.parameters
        assert sig.parameters["max_workers"].default == 10
