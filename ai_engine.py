"""
AI Meaning Engine — used in Meaning Mode (toggle button on, momentary off).

Flow (new):
1. Gesture Engine finds pointed word + sentence context (current page).
2. Word and context are cleaned against Septopus vocabulary.
3. Groq explains TWO things in one pass:
     (a) what the pointed word means in this sentence, and
     (b) WHY the character is saying this line, using the merged memory
         of everything read so far (never anything beyond the pointer).
"""
import json
import logging
from typing import Dict, Optional

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
- CRITICAL: Output strictly in English only. NEVER translate to or output any other language.
- Keep the explanation to 1 or 2 concise, clear sentences in simple, friendly English.
- Plain, friendly language suitable for young readers.
- Explain the specific meaning of the word in this sentence.
- Do NOT repeat the sentence verbatim.
- Output ONLY the explanation text in English, no markdown or preamble.
- And give an example suited for dyslexic people. """

# ---------------------------------------------------------------------------
# NEW: combined meaning + situational-context prompt
# ---------------------------------------------------------------------------
CONTEXT_SYSTEM = """You are a reading companion for the book "Septopus: Trouble on the High Cs" \
by Jyotin Goel, helping a young reader who has just pointed their finger at a specific word on the page.

You will receive THREE pieces of information:
  1. TARGET WORD: the exact word the reader pointed at.
  2. CURRENT LINE: the sentence on the page that contains the target word, and what \
the character is currently saying or doing.
  3. STORY SO FAR: a slice of everything the reader has already read up to this point \
in the book (never anything beyond this).

Your job is to write a SHORT response — 3 to 4 lines maximum — that does BOTH of:
  (a) Explains what the TARGET WORD means, using simple, friendly language suitable \
for a young reader.
  (b) Explains WHY the character is saying this line right now, drawing on the STORY SO FAR.

STRICT RULES (do not break any of these):
- ABSOLUTELY NO SPOILERS. Only use facts that appear in STORY SO FAR or CURRENT LINE. \
Never invent or reference anything that happens later in the book. If the story hasn't \
told the reader why yet, say so honestly (e.g. "We don't know yet — let's read on").
- Output strictly in English only. Never translate to any other language.
- Plain, friendly, accessible English for young readers.
- Do NOT repeat the CURRENT LINE verbatim.
- Do NOT use markdown, bullet points, asterisks, headers, or greetings.
- Do NOT add a preamble like "Sure, here's..." or "The word means...". Just answer.
- Separate the word meaning and the situational context with a single newline so the \
OLED can render them as two short paragraphs.
- Keep the whole response under 220 characters so it fits the display.
"""


def clean_word_and_context(word: str, context: str) -> Dict[str, str]:
    if not word or not word.strip():
        return {"word": "", "context": context}
    clean_w = word.strip().strip(".,!?;:\"'()").strip()
    # Fast path: if word is clean English characters/digits without OCR artifacts, skip redundant LLM call
    if clean_w.replace("-", "").isalnum() and len(clean_w) <= 18:
        return {"word": clean_w, "context": context}

    try:
        raw = call_merge_engine(
            WORD_CLEANUP_SYSTEM,
            json.dumps({"word": word, "context": context}),
            json_mode=True,
        )
        return json.loads(raw)
    except Exception as e:
        log.warning("Word/context cleanup failed, using raw values: %s", e)
        return {"word": clean_w, "context": context}


def _is_mostly_english(text: str) -> bool:
    """Verifies text contains only standard English/Latin characters."""
    if not text:
        return False
    non_ascii_count = sum(1 for c in text if ord(c) > 127 and c not in "“”‘’—–…")
    return (non_ascii_count / len(text)) < 0.1


# ---------------------------------------------------------------------------
# Existing: meaning-only. Kept for the two-consecutive-words rule.
# ---------------------------------------------------------------------------
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
        user_prompt = (
            f'Book: Septopus: Trouble on the High Cs\n'
            f'Target Word: "{w}"\n'
            f'Sentence Context: "{ctx}"\n'
            f'Language: English ONLY'
        )
        explanation = call_ai_engine(MEANING_SYSTEM, user_prompt).strip()

        if not explanation or not _is_mostly_english(explanation):
            log.warning("Discarded non-English or empty explanation for '%s'", w)
            return ""

        log.info("Meaning for '%s': %s", w, explanation)
        return explanation
    except Exception as e:
        log.error("AI engine meaning lookup failed for '%s': %s", w, e)
        return ""


# ---------------------------------------------------------------------------
# NEW: meaning + why-the-character-is-saying-this, grounded in merged memory
# ---------------------------------------------------------------------------
def explain_with_context(
    word: str,
    current_line: str,
    memory_text: str,
    max_memory_words: int = 1200,
) -> str:
    """
    Meaning-mode entry point that returns BOTH:
      - what the pointed word means, and
      - why the character is saying this line, using the merged memory.

    Args:
        word:          The cleaned target word the reader pointed at.
        current_line:  The sentence on the page containing that word (from OCR).
        memory_text:   Everything read so far, as one string. The caller must
                       slice this so it only contains text UP TO the reading
                       pointer — never beyond it. This is what enforces the
                       no-spoiler rule at the code level.
        max_memory_words: Trim memory to this many trailing words to keep the
                       Groq prompt cheap and fast.

    Returns:
        A short display-ready string (two short paragraphs separated by \n),
        or "" on failure.
    """
    if not word or not word.strip():
        return ""
    if not current_line or not current_line.strip():
        return ""

    cleaned = clean_word_and_context(word, current_line)
    w = cleaned.get("word", word).strip()
    ctx = cleaned.get("context", current_line).strip()

    # ---- Trim memory slice to most recent N words ----
    memory_slice = (memory_text or "").strip()
    if memory_slice:
        words = memory_slice.split()
        if len(words) > max_memory_words:
            memory_slice = " ".join(words[-max_memory_words:])
    else:
        memory_slice = "(The reader has just started — no prior story context yet.)"

    user_prompt = (
        f'Book: Septopus: Trouble on the High Cs by Jyotin Goel\n\n'
        f'TARGET WORD:\n"{w}"\n\n'
        f'CURRENT LINE (what the character is saying right now):\n"{ctx}"\n\n'
        f'STORY SO FAR (only what the reader has already read — never spoiler beyond this):\n'
        f'"""\n{memory_slice}\n"""\n\n'
        f'Write 3-4 lines: first the meaning of "{w}", then why the character is saying this now.\n'
        f'Language: English ONLY.'
    )

    try:
        response = call_ai_engine(CONTEXT_SYSTEM, user_prompt).strip()

        if not response or not _is_mostly_english(response):
            log.warning("Discarded non-English or empty contextual response for '%s'", w)
            return ""

        # Enforce a hard length cap so the OLED doesn't overflow
        if len(response) > 220:
            response = response[:217].rsplit(" ", 1)[0] + "..."

        log.info("Contextual explanation for '%s': %s", w, response.replace("\n", " | "))
        return response
    except GroqError as e:
        log.error("Groq error during contextual explanation for '%s': %s", w, e)
        return ""
    except Exception as e:
        log.error("Contextual explanation failed for '%s': %s", w, e)
        return ""


# ---------------------------------------------------------------------------
# Recap (unchanged)
# ---------------------------------------------------------------------------
_meaning_cache: Dict[str, str] = {}

RECAP_SYSTEM = """You summarize reading progress for the book "Septopus: Trouble on the High Cs" by Jyotin Goel.
Rules:
1. Summarize what has happened in the story so far in 4 to 5 short, friendly lines.
2. STRICTLY DO NOT mention or invent anything that happens beyond the provided text slice (STRICTLY NO SPOILERS).
3. Output strictly in plain, accessible English suitable for young readers.
4. Output ONLY the summary lines. Do NOT include greetings, intro phrases, bullet points, asterisks, or markdown formatting."""


def meaning_of(word: str, context: str = "") -> str:
    """
    Returns a normalized (lowercased, stripped) meaning string for stable equality checks,
    caching results to avoid redundant API queries.
    """
    if not word or not word.strip():
        return ""
    norm_w = word.strip().strip(".,!?;:\"'()").lower()
    if not norm_w:
        return ""
    if norm_w in _meaning_cache:
        return _meaning_cache[norm_w]

    raw_exp = explain_word(word, context)
    norm_exp = raw_exp.strip().lower()
    _meaning_cache[norm_w] = norm_exp
    return norm_exp


def generate_recap(text_slice: str, max_lines: int = 5) -> str:
    """
    Calls Groq to generate a 4-5 line recap of everything read so far.
    Guarantees no spoilers since only past read text is provided in the prompt.
    """
    clean_slice = text_slice.strip()
    if not clean_slice:
        return "You have just started reading. No story events have taken place yet."

    words = clean_slice.split()
    if len(words) > 3000:
        clean_slice = " ".join(words[-3000:])

    prompt = f"""Book: Septopus: Trouble on the High Cs by Jyotin Goel
Story content read up to this moment (DO NOT SPOIL BEYOND THIS POINT):
\"\"\"{clean_slice}\"\"\"

Please provide a concise recap of what has happened so far in {max_lines} lines."""

    try:
        recap = call_ai_engine(RECAP_SYSTEM, prompt).strip()
        if not recap or not _is_mostly_english(recap):
            log.warning("Recap returned empty or non-English text.")
            return "Unable to generate recap at this time."
        log.info("Generated recap: %s", recap.replace("\n", " "))
        return recap
    except Exception as e:
        log.error("AI engine recap generation failed: %s", e)
        return "Could not load story recap. Please try again."


# ---------------------------------------------------------------------------
# NEW: Intent mode entry point (explaining why character says this line)
# ---------------------------------------------------------------------------
INTENT_CONTEXT_SYSTEM = """You are a reading companion for the book "Septopus: Trouble on the High Cs" \
by Jyotin Goel. A young reader has tapped an intent button while pointing at a specific word on the page.

You will receive THREE pieces of information:
  1. TARGET WORD: the word the reader is pointing at.
  2. CURRENT LINE: the sentence on the page that contains the target word — what the \
character is saying or doing right now.
  3. STORY SO FAR: everything the reader has already read up to this point. Never anything beyond this.

Write a SHORT response (3-4 lines maximum, under 220 characters) that gives the reader \
CONTEXT — specifically:
  - Why is the character saying this line right now? What led up to this moment?
  - If the target word is unusual or important, briefly explain what it means in this scene.

STRICT RULES:
- ABSOLUTELY NO SPOILERS. Use only facts present in STORY SO FAR or CURRENT LINE. If the \
story has not yet revealed why, say so honestly — "we'll find out soon" or "the story hasn't \
told us yet." Never invent.
- Output strictly in English only.
- Plain, friendly language for young readers.
- Do NOT repeat the CURRENT LINE verbatim.
- No markdown, no bullets, no greetings, no preamble.
- Separate the "why now" explanation and the "what it means" explanation with a single \
newline so the OLED renders two short paragraphs.
"""


def explain_intent_context(
    word: str,
    current_line: str,
    memory_text: str,
    max_memory_words: int = 1200,
) -> str:
    """
    Intent-mode entry point: explains WHY the character is saying this line,
    grounded in everything read so far. No spoilers — caller must slice memory
    up to the reading pointer before calling.
    """
    if not word or not word.strip():
        return ""
    if not current_line or not current_line.strip():
        return ""

    cleaned = clean_word_and_context(word, current_line)
    w = cleaned.get("word", word).strip()
    ctx = cleaned.get("context", current_line).strip()

    memory_slice = (memory_text or "").strip()
    if memory_slice:
        words = memory_slice.split()
        if len(words) > max_memory_words:
            memory_slice = " ".join(words[-max_memory_words:])
    else:
        memory_slice = "(The reader has just started — no prior story context yet.)"

    user_prompt = (
        f'Book: Septopus: Trouble on the High Cs by Jyotin Goel\n\n'
        f'TARGET WORD:\n"{w}"\n\n'
        f'CURRENT LINE:\n"{ctx}"\n\n'
        f'STORY SO FAR (only up to the reading pointer — no spoilers):\n'
        f'"""\n{memory_slice}\n"""\n\n'
        f'Explain why this line is being said right now and what it means.\n'
        f'Language: English ONLY.'
    )

    try:
        response = call_ai_engine(INTENT_CONTEXT_SYSTEM, user_prompt).strip()
        if not response or not _is_mostly_english(response):
            log.warning("Intent context returned empty/non-English for '%s'", w)
            return ""
        if len(response) > 220:
            response = response[:217].rsplit(" ", 1)[0] + "..."
        log.info("Intent context for '%s': %s", w, response.replace("\n", " | "))
        return response
    except Exception as e:
        log.error("Intent context failed for '%s': %s", w, e)
        return ""

