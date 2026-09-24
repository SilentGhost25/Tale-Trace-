"""
OCR Dynamic Memory Processor — handles semantic page checks,
Groq-based text merging, and reading pointer calculation.

Adapted from OCRandGESTURE for the Logitech C270 book-reading companion.
"""
from difflib import SequenceMatcher
import logging
import os
import sys
from typing import Optional, List, Tuple

from groq import Groq

from config import settings

log = logging.getLogger("taletrace.memory")

# Domain knowledge prompt for Septopus: Trouble on the High Cs by Jyotin Goel
SEPTOPUS_SYSTEM_PROMPT = """
You are the Memory Merge Engine for the TaleTrace reading system, reading the children's book:
"Septopus: Trouble on the High Cs" by Jyotin Goel.

Key book characters and terms (DO NOT "correct" these creative names):
- Rot8: the brave octopus protagonist with seven-and-a-half tentacles.
- Tumboo: Rot8's loyal turtle friend.
- The Oct-estra: the famous octopus musical orchestra.
- Band members: Irrit8, Imit8, Emul8, Po8, Vibr8.
- Zubair Midha: the world-famous music maestro and conductor.
- Jai Kalia: the scheming millionaire villain.

Your SOLE responsibility is reconstructing and accumulating raw OCR fragments into a single, seamless, complete page text.

CRITICAL MERGE RULES:
1. ACCUMULATE & EXPAND: If 'Current Merge Memory' has the top of the page, and 'New OCR Output' captures the middle or bottom of the page, STITCH THEM TOGETHER into one complete text. NEVER throw away the top of the page.
2. OVERLAP DEDUPLICATION: Use the overlapping words/sentences in the middle to seamlessly stitch the two portions together and eliminate duplicate phrases.
3. REMOVE NOISE: Remove running headers (author name "Jyotin Goel", book title "Septopus: Trouble on the High Cs") and standalone page numbers.
4. REPAIR OCR ERRORS: Smoothly repair noisy OCR words using contextual understanding, while strictly preserving book character names (Rot8, Tumboo, Oct-estra, etc.).
5. NO CONVERSATION: DO NOT summarize, converse, or add markdown code blocks. Return ONLY the complete merged book text.
6. ENGLISH ONLY: Output strictly in standard English only.
"""


def _get_groq_client() -> Optional[Groq]:
    api_key = settings.groq_key_merge or settings.groq_api_key
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        log.warning("Could not initialize Groq client: %s", e)
        return None


def merge_ocr_with_groq(current_merge_memory: str, new_ocr_text: str) -> str:
    """Groq reconstructs and accumulates overlapping text fragments into clean, complete page text."""
    clean_new = new_ocr_text.strip()
    if not clean_new:
        return current_merge_memory

    # If merge memory already contains this exact or near-exact text, avoid redundant API calls
    if current_merge_memory:
        if clean_new in current_merge_memory:
            return current_merge_memory
        # Quick token overlap check
        mem_tokens = set(current_merge_memory.lower().split())
        new_tokens = set(clean_new.lower().split())
        if new_tokens and len(new_tokens.intersection(mem_tokens)) / len(new_tokens) > 0.95:
            return current_merge_memory

    user_prompt = f"""
Current Accumulated Memory (Top / Earlier text from page):
{current_merge_memory if current_merge_memory else "[EMPTY - PAGE START]"}

New Camera Capture (Latest OCR frame):
{clean_new}

Combined Cumulative Text (Seamlessly stitched with middle overlap deduplicated, keeping ALL text):
"""
    client = _get_groq_client()
    if client is None:
        log.warning("Groq API key not set — appending raw OCR text.")
        return (current_merge_memory + "\n" + clean_new).strip() if current_merge_memory else clean_new

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SEPTOPUS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=settings.groq_model,
            temperature=0.1,
        )
        res = completion.choices[0].message.content.strip()
        # Sanity check: merged text should not lose previous memory unless intentional
        if res and len(res.split()) >= len(current_merge_memory.split()) * 0.7:
            return res
        return (current_merge_memory + "\n" + clean_new).strip() if current_merge_memory else clean_new
    except Exception as e:
        log.error("Groq text merge error: %s", e)
        return (current_merge_memory + "\n" + clean_new).strip() if current_merge_memory else clean_new


class TaleTraceMemoryManager:
    """
    Manages active page reading memory, page transition detection,
    and reading pointer synchronization.
    """

    def __init__(self):
        self.merge_memory = ""          # Active page text
        self.permanent_memory: List[str] = []  # Archived completed pages
        self.current_reading_pointer = 0
        self.current_word_index = 0

    def check_same_page_via_groq(self, active_mem: str, raw_ocr: str) -> bool:
        """Determines whether the raw OCR frame belongs to the current page by checking overlap."""
        if not active_mem.strip() or not raw_ocr.strip():
            return True

        # Fast local lexical overlap check across ALL words in active memory
        mem_words = set(w.strip(".,!?;:\"'()").lower() for w in active_mem.split() if len(w) > 2)
        ocr_words = set(w.strip(".,!?;:\"'()").lower() for w in raw_ocr.split() if len(w) > 2)
        if mem_words and ocr_words:
            overlap = len(mem_words.intersection(ocr_words))
            # Even a few common words (top/bottom overlap) indicates the same page
            if overlap >= 2 or (overlap / len(ocr_words)) >= 0.15:
                return True

        client = _get_groq_client()
        if client is None:
            return True

        prompt = f"""Analyze these two text blocks from a reading camera stream:

ACCUMULATED PAGE MEMORY:
"{active_mem[-400:]}"

LATEST CAPTURED FRAME:
"{raw_ocr[:400]}"

Are both texts from the SAME page, scene, or continuous reading session (with overlapping or continuous lines)?
Answer with strictly ONE word: YES or NO."""

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.groq_fast_model,
                temperature=0.0,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            log.warning("Groq similarity check failed: %s; defaulting to same page.", e)
            return True

    def calculate_accurate_pointer(self, old_text: str, updated_merged_text: str) -> int:
        """Calculates where insertion occurred to preserve the TTS reading pointer."""
        if not old_text:
            return 0
        matcher = SequenceMatcher(None, old_text, updated_merged_text)
        match = matcher.find_longest_match(0, len(old_text), 0, len(updated_merged_text))
        return match.b + match.size

    def process_ocr_text(self, latest_ocr_text: str) -> str:
        """Merges new OCR frame into active memory, updating pointers."""
        if not latest_ocr_text.strip():
            return self.merge_memory

        is_same_page = True
        if self.merge_memory:
            is_same_page = self.check_same_page_via_groq(self.merge_memory, latest_ocr_text)

        if not is_same_page:
            log.info("Page transition detected. Archiving previous page to permanent memory.")
            self.commit_to_permanent_memory()
            self.current_reading_pointer = 0
            self.current_word_index = 0

        reconstructed = merge_ocr_with_groq(self.merge_memory, latest_ocr_text)

        if is_same_page and self.merge_memory:
            self.current_reading_pointer = self.calculate_accurate_pointer(
                self.merge_memory, reconstructed
            )

        self.merge_memory = reconstructed
        return self.merge_memory

    def commit_to_permanent_memory(self):
        if self.merge_memory.strip():
            self.permanent_memory.append(self.merge_memory)
            log.info("Committed page to permanent memory. Total pages: %d", len(self.permanent_memory))
        self.merge_memory = ""


pipeline = TaleTraceMemoryManager()
