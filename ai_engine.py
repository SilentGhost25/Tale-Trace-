"""
AI Meaning Engine — used in Meaning Mode (toggle button on, momentary off).

Flow:
1. Gesture Engine finds pointed word + sentence context.
2. Word and context are cleaned against Septopus: Trouble on the High Cs vocabulary.
3. Groq explains the word's meaning in that specific context (1-2 clear, reader-friendly sentences).
"""
import json
import logging
from typing import Dict

from groq_clients import call_merge_engine, call_ai_engine, GroqError

log = logging.getLogger("taletrace.ai")

WORD_CLEANUP_SYSTEM = """You clean up an OCR-captured word and sentence context from the book \
"Septopus: Trouble on the High Cs" by Jyotin Goel.
Key book terms: Rot8 (octopus hero), Tumboo (turtle), Oct-estra (octopus band), \
Irrit8, Imit8, Emul8, Po8, Vibr8, Zubair Midha, Jai Kalia.
Fix OCR misspellings in the target word and sentence context, but preserve book character names.
CRITICAL: All output MUST be strictly in English only. NEVER output any foreign language.
Return ONLY JSON: {"word": "...", "context": "..."}"""

MEANING_SYSTEM = """You explain what a word means as used in the book "Septopus: Trouble on the High Cs" \
for a reader who pointed to it while reading aloud.
Rules:
- CRITICAL: Output strictly in English only. NEVER translate to or output any other language (e.g., Hindi, Spanish, French, etc.).
- Keep the explanation to 1 or 2 concise, clear sentences in simple, friendly English.
- Plain, friendly language suitable for young readers.
- Explain the specific meaning of the word in this sentence.
- Do NOT repeat the sentence verbatim.
- Output ONLY the explanation text in English, no markdown or preamble.
- And give an example suited for dyslexic people. """


def clean_word_and_context(word: str, context: str) -> Dict[str, str]:
    if not word:
        return {"word": "", "context": context}
    try:
        raw = call_merge_engine(
            WORD_CLEANUP_SYSTEM,
            json.dumps({"word": word, "context": context}),
            json_mode=True,
        )
        return json.loads(raw)
    except Exception as e:
        log.warning("Word/context cleanup failed, using raw values: %s", e)
        return {"word": word, "context": context}


def _is_mostly_english(text: str) -> bool:
    """Verifies text contains only standard English/Latin characters."""
    if not text:
        return False
    # Check that non-ascii characters are minimal (e.g. curly quotes/dashes allowed)
    non_ascii_count = sum(1 for c in text if ord(c) > 127 and c not in "“”‘’—–…")
    return (non_ascii_count / len(text)) < 0.1


def explain_word(word: str, context: str) -> str:
    """
    Returns a short, display-ready meaning string for the OLED display.
    If unable to fetch a definition, returns an empty string so nothing is displayed.
    """
    if not word or not word.strip():
        return ""

    cleaned = clean_word_and_context(word, context)
    w = cleaned.get("word", word).strip()
    ctx = cleaned.get("context", context).strip()

    try:
        user_prompt = f'Book: Septopus: Trouble on the High Cs\nTarget Word: "{w}"\nSentence Context: "{ctx}"\nLanguage: English ONLY'
        explanation = call_ai_engine(MEANING_SYSTEM, user_prompt).strip()

        if not explanation or not _is_mostly_english(explanation):
            log.warning("Discarded non-English or empty explanation for '%s'", w)
            return ""

        log.info("Meaning for '%s': %s", w, explanation)
        return explanation
    except Exception as e:
        log.error("AI engine meaning lookup failed for '%s': %s", w, e)
        return ""

