"""
Argos Translate module for offline translation.

Uses Argos Translate for fully offline, open-source translation.
Supports:
- Direct translation (e.g., ar -> en)
- Automatic pivot via English when direct pair is unavailable (e.g., ar -> en -> id)
- Explicit pivot mode for maximum control

Language pairs commonly used:
- ar -> en  (Arabic to English) — direct
- ar -> id  (Arabic to Indonesian) — typically via English pivot (ar -> en -> id)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SETUP_DONE = set()          # track which language pairs have been set up
_PIVOT_PAIRS = set()         # pairs that require pivoting through English
_DIRECT_PAIRS = set()        # pairs that have direct translation available


def _get_installed_pair_codes() -> set:
    """Return a set of (from_code, to_code) tuples for installed packages."""
    import argostranslate.package
    installed = argostranslate.package.get_installed_packages()
    return {(p.from_code, p.to_code) for p in installed if p}


def check_direct_available(from_code: str, to_code: str) -> bool:
    """
    Check if a direct translation package is available (installed or downloadable).

    Returns:
        True if direct translation is possible, False if pivot is needed.
    """
    pair_key = f"{from_code}->{to_code}"
    if pair_key in _DIRECT_PAIRS:
        return True
    if pair_key in _PIVOT_PAIRS:
        return False

    # Check installed packages first
    try:
        installed = _get_installed_pair_codes()
        if (from_code, to_code) in installed:
            _DIRECT_PAIRS.add(pair_key)
            return True
    except Exception:
        pass

    # Check available packages
    try:
        import argostranslate.package
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        for p in available:
            if p and p.from_code == from_code and p.to_code == to_code:
                _DIRECT_PAIRS.add(pair_key)
                return True
    except Exception:
        pass

    return False


def setup_argos_translation(from_code: str = "ar", to_code: str = "en") -> str:
    """
    Ensure Argos translation packages are installed for the given language pair.

    Tries to find a direct package (from_code -> to_code). If not found,
    falls back to installing intermediate packages via English
    (from_code -> en, and en -> to_code).

    Args:
        from_code: Source language code (default: 'ar' for Arabic).
        to_code: Target language code (default: 'en' for English).

    Returns:
        Translation route description string:
        - "direct" if direct translation is available
        - "pivot:en" if using English as intermediate
    """
    pair_key = f"{from_code}->{to_code}"
    if pair_key in _SETUP_DONE:
        route = "direct" if pair_key in _DIRECT_PAIRS else "pivot:en"
        logger.debug("Argos translation already set up for %s (%s)", pair_key, route)
        return route

    import argostranslate.package
    import argostranslate.translate

    logger.info("Setting up Argos translation (%s -> %s)...", from_code, to_code)
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Filter out None values
    valid_packages = [p for p in available_packages if p]

    def find_package(f_code, t_code):
        try:
            return next(
                p for p in valid_packages
                if p.from_code == f_code and p.to_code == t_code
            )
        except StopIteration:
            return None

    # 1. Try direct translation package
    direct_package = find_package(from_code, to_code)
    if direct_package:
        logger.info("Installing direct translation package: %s -> %s", from_code, to_code)
        argostranslate.package.install_from_path(direct_package.download())
        _SETUP_DONE.add(pair_key)
        _DIRECT_PAIRS.add(pair_key)
        logger.info("Direct translation ready: %s -> %s", from_code, to_code)
        return "direct"

    # 2. Fallback to intermediate translation via English
    if from_code != "en" and to_code != "en":
        logger.info(
            "No direct package for %s -> %s. Setting up pivot via English.",
            from_code, to_code,
        )
        from_en_package = find_package(from_code, "en")
        en_to_package = find_package("en", to_code)

        if from_en_package and en_to_package:
            logger.info("Installing: %s -> en", from_code)
            argostranslate.package.install_from_path(from_en_package.download())

            logger.info("Installing: en -> %s", to_code)
            argostranslate.package.install_from_path(en_to_package.download())

            _SETUP_DONE.add(pair_key)
            _PIVOT_PAIRS.add(pair_key)
            logger.info(
                "Pivot translation ready: %s -> en -> %s", from_code, to_code
            )
            return "pivot:en"

    # 3. If no path found, raise an error
    raise RuntimeError(
        f"Could not find a translation path for {from_code} -> {to_code}. "
        f"Neither a direct package nor an intermediate path via 'en' was found. "
        f"Available language pairs: {[(p.from_code, p.to_code) for p in valid_packages[:20]]}"
    )


def translate_text(
    text: str,
    from_code: str = "ar",
    to_code: str = "en",
    pivot_through: Optional[str] = None,
) -> str:
    """
    Translate text using Argos Translate (fully offline).

    Supports three modes:
    1. Direct translation (from_code -> to_code) — used when direct package exists
    2. Auto-pivot — if direct fails, automatically tries from_code -> en -> to_code
    3. Explicit pivot — if pivot_through is set, always uses two-step translation

    Args:
        text: Text to translate.
        from_code: Source language code (default: 'ar').
        to_code: Target language code (default: 'en').
        pivot_through: If set, always pivot through this language (e.g., 'en').
            Useful for ar -> id where you want: ar -> en -> id.

    Returns:
        Translated text string. Returns empty string for empty input.
        Returns original text if all translation attempts fail.
    """
    if not text or not text.strip():
        return ""

    import argostranslate.translate

    pair_key = f"{from_code}->{to_code}"

    # Explicit pivot mode: always go through the specified intermediate language
    if pivot_through and from_code != pivot_through and to_code != pivot_through:
        logger.debug(
            "Pivot translation: %s -> %s -> %s",
            from_code, pivot_through, to_code,
        )
        try:
            intermediate = argostranslate.translate.translate(
                text, from_code, pivot_through
            )
            result = argostranslate.translate.translate(
                intermediate, pivot_through, to_code
            )
            return result
        except Exception as e:
            logger.warning(
                "Explicit pivot translation failed (%s -> %s -> %s): %s",
                from_code, pivot_through, to_code, e,
            )
            return text

    # Known pivot pair: use two-step translation
    if pair_key in _PIVOT_PAIRS:
        logger.debug("Using known pivot route: %s -> en -> %s", from_code, to_code)
        try:
            en_text = argostranslate.translate.translate(text, from_code, "en")
            result = argostranslate.translate.translate(en_text, "en", to_code)
            return result
        except Exception as e:
            logger.warning("Pivot translation failed: %s", e)
            # Try direct as fallback
            try:
                return argostranslate.translate.translate(text, from_code, to_code)
            except Exception:
                return text

    # Direct translation (default path)
    try:
        translated = argostranslate.translate.translate(text, from_code, to_code)
        return translated
    except Exception as e:
        logger.warning("Direct translation failed for '%s...': %s", text[:50], e)

        # Auto-pivot through English if direct fails and target isn't English
        if from_code != "en" and to_code != "en":
            logger.info("Attempting auto-pivot: %s -> en -> %s", from_code, to_code)
            try:
                en_text = argostranslate.translate.translate(text, from_code, "en")
                result = argostranslate.translate.translate(en_text, "en", to_code)
                # Remember this pair needs pivoting for future calls
                _PIVOT_PAIRS.add(pair_key)
                return result
            except Exception as e2:
                logger.error("Auto-pivot translation also failed: %s", e2)

        return text  # Return original text as fallback


def get_supported_languages() -> list:
    """
    Get list of available language codes from Argos Translate.

    Returns:
        List of dicts with 'code' and 'name' for each language.
    """
    try:
        import argostranslate.translate
        languages = argostranslate.translate.get_installed_languages()
        return [{"code": lang.code, "name": lang.name} for lang in languages]
    except Exception:
        return []


def get_translation_route(from_code: str, to_code: str) -> str:
    """
    Determine the translation route for a language pair.

    Returns:
        "direct" — direct package available
        "pivot:en" — needs English as intermediate
        "unavailable" — no translation path found
    """
    pair_key = f"{from_code}->{to_code}"

    if pair_key in _DIRECT_PAIRS:
        return "direct"
    if pair_key in _PIVOT_PAIRS:
        return "pivot:en"
    if from_code == to_code:
        return "direct"

    # Check if direct is available
    if check_direct_available(from_code, to_code):
        return "direct"

    # Check if pivot is possible
    if from_code != "en" and to_code != "en":
        if (check_direct_available(from_code, "en")
                and check_direct_available("en", to_code)):
            return "pivot:en"

    return "unavailable"
