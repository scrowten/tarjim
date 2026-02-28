"""
Tests for the tarjim Argos translator module.

Tests cover:
- Empty input handling
- Direct translation calls
- Pivot translation (ar → en → id)
- Auto-pivot fallback when direct fails
- Explicit pivot_through parameter
- Translation route detection
- Setup caching
- Error fallback behavior

Note: These tests mock Argos to avoid downloading actual translation packages.
For integration tests that verify real translation quality, run separately.
"""

from unittest.mock import patch, MagicMock, call
import pytest

from src.core import translator_argos


# ===========================
# Fixtures / helpers
# ===========================

@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all module-level caches before each test."""
    translator_argos._SETUP_DONE.clear()
    translator_argos._PIVOT_PAIRS.clear()
    translator_argos._DIRECT_PAIRS.clear()
    yield
    translator_argos._SETUP_DONE.clear()
    translator_argos._PIVOT_PAIRS.clear()
    translator_argos._DIRECT_PAIRS.clear()


def _make_fake_argos_module(translate_fn):
    """Create a fake argostranslate.translate module with the given translate function."""
    fake_module = MagicMock()
    fake_module.translate = translate_fn
    return fake_module


# ===========================
# Test: Empty input
# ===========================

def test_translate_text_empty_input():
    """translate_text should return empty string for empty input."""
    assert translator_argos.translate_text("") == ""
    assert translator_argos.translate_text("   ") == ""
    assert translator_argos.translate_text(None) == ""


# ===========================
# Test: Direct translation
# ===========================

def test_translate_text_direct_calls_argos():
    """translate_text should call argostranslate.translate.translate for direct pairs."""
    mock_translate = MagicMock(return_value="Hello")
    fake_module = _make_fake_argos_module(mock_translate)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text("مرحبا", from_code="ar", to_code="en")
        assert result == "Hello"
        mock_translate.assert_called_once_with("مرحبا", "ar", "en")


# ===========================
# Test: Pivot translation (known pivot pair)
# ===========================

def test_translate_text_uses_pivot_for_known_pivot_pair():
    """When a pair is in _PIVOT_PAIRS, translate_text should route through English."""
    translator_argos._PIVOT_PAIRS.add("ar->id")

    call_log = []

    def mock_translate(text, from_code, to_code):
        call_log.append((text, from_code, to_code))
        if from_code == "ar" and to_code == "en":
            return "Hello"
        if from_code == "en" and to_code == "id":
            return "Halo"
        return text

    fake_module = _make_fake_argos_module(mock_translate)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text("مرحبا", from_code="ar", to_code="id")
        assert result == "Halo"
        # Should have called ar->en first, then en->id
        assert len(call_log) == 2
        assert call_log[0] == ("مرحبا", "ar", "en")
        assert call_log[1] == ("Hello", "en", "id")


# ===========================
# Test: Explicit pivot_through parameter
# ===========================

def test_translate_text_explicit_pivot_through():
    """translate_text with pivot_through='en' should always pivot, regardless of _PIVOT_PAIRS."""
    call_log = []

    def mock_translate(text, from_code, to_code):
        call_log.append((text, from_code, to_code))
        if from_code == "ar" and to_code == "en":
            return "Peace"
        if from_code == "en" and to_code == "id":
            return "Damai"
        return text

    fake_module = _make_fake_argos_module(mock_translate)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text(
            "سلام", from_code="ar", to_code="id", pivot_through="en"
        )
        assert result == "Damai"
        assert len(call_log) == 2
        assert call_log[0] == ("سلام", "ar", "en")
        assert call_log[1] == ("Peace", "en", "id")


def test_translate_text_explicit_pivot_skipped_when_same_as_source():
    """pivot_through should be skipped if it equals from_code."""
    mock_translate = MagicMock(return_value="Translated")
    fake_module = _make_fake_argos_module(mock_translate)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        # from_code == pivot_through → should go direct
        result = translator_argos.translate_text(
            "hello", from_code="en", to_code="id", pivot_through="en"
        )
        assert result == "Translated"
        mock_translate.assert_called_once_with("hello", "en", "id")


# ===========================
# Test: Auto-pivot fallback
# ===========================

def test_translate_text_auto_pivot_on_direct_failure():
    """When direct translation fails for non-English target, auto-pivot through English."""
    call_count = [0]

    def mock_translate(text, from_code, to_code):
        call_count[0] += 1
        # First call: direct ar->id fails
        if from_code == "ar" and to_code == "id":
            raise RuntimeError("No direct ar->id package")
        # Pivot calls succeed
        if from_code == "ar" and to_code == "en":
            return "Hello"
        if from_code == "en" and to_code == "id":
            return "Halo"
        return text

    fake_module = _make_fake_argos_module(mock_translate)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text("مرحبا", from_code="ar", to_code="id")
        assert result == "Halo"
        # Should have been 3 calls: direct (fail), pivot ar->en, pivot en->id
        assert call_count[0] == 3
        # Should now remember this pair needs pivoting
        assert "ar->id" in translator_argos._PIVOT_PAIRS


# ===========================
# Test: Fallback to original text
# ===========================

def test_translate_text_returns_original_when_all_fails():
    """translate_text should return original text if all translation attempts fail."""
    def always_fail(*args, **kwargs):
        raise RuntimeError("Translation unavailable")

    fake_module = _make_fake_argos_module(always_fail)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text("مرحبا", from_code="ar", to_code="en")
        assert result == "مرحبا"


def test_translate_text_returns_original_when_pivot_also_fails():
    """Even if pivot fails, should return original text, not crash."""
    def always_fail(*args, **kwargs):
        raise RuntimeError("All translation broken")

    fake_module = _make_fake_argos_module(always_fail)

    with patch.dict('sys.modules', {
        'argostranslate': MagicMock(),
        'argostranslate.translate': fake_module,
    }):
        result = translator_argos.translate_text("مرحبا", from_code="ar", to_code="id")
        assert result == "مرحبا"


# ===========================
# Test: Setup caching
# ===========================

def test_setup_argos_caches_pair():
    """setup_argos_translation should return cached route for already-setup pairs."""
    translator_argos._SETUP_DONE.add("ar->en")
    translator_argos._DIRECT_PAIRS.add("ar->en")

    # Should return immediately without hitting Argos
    route = translator_argos.setup_argos_translation(from_code="ar", to_code="en")
    assert route == "direct"


def test_setup_argos_caches_pivot_pair():
    """setup_argos_translation should return 'pivot:en' for cached pivot pairs."""
    translator_argos._SETUP_DONE.add("ar->id")
    translator_argos._PIVOT_PAIRS.add("ar->id")

    route = translator_argos.setup_argos_translation(from_code="ar", to_code="id")
    assert route == "pivot:en"


# ===========================
# Test: get_translation_route
# ===========================

def test_get_translation_route_known_direct():
    """get_translation_route returns 'direct' for known direct pairs."""
    translator_argos._DIRECT_PAIRS.add("ar->en")
    assert translator_argos.get_translation_route("ar", "en") == "direct"


def test_get_translation_route_known_pivot():
    """get_translation_route returns 'pivot:en' for known pivot pairs."""
    translator_argos._PIVOT_PAIRS.add("ar->id")
    assert translator_argos.get_translation_route("ar", "id") == "pivot:en"


def test_get_translation_route_same_language():
    """get_translation_route returns 'direct' when source == target."""
    assert translator_argos.get_translation_route("ar", "ar") == "direct"
