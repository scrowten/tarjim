import translators as ts

def translate_text(text: str, target_language: str = 'id') -> str:
    """
    Translates a given text to the target language.
    Includes a retry mechanism with a different translation service if the default fails.
    """
    if not text or not text.strip():
        return ""
    try:
        # Using the 'google' translator by default
        return ts.translate_text(text, translator='google', to_language=target_language)
    except Exception as e:
        print(f"Google translator failed: {e}. Trying with 'bing'.")
        try:
            # Fallback to 'bing' translator
            return ts.translate_text(text, translator='bing', to_language=target_language)
        except Exception as e2:
            print(f"Bing translator also failed: {e2}. Returning original text.")
            return text # Return original text if all translators fail
