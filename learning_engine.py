"""
Learning Engine — not part of the four button states, but wired in since the
.env explicitly reserves GROQ_API_KEY_3 for it: generates a short quiz from
the session's reading memory + the words the reader asked about, once a
session ends. Call end_of_session_quiz() from main.py's shutdown path.
"""
import json
import logging

from groq_clients import call_learning_engine, GroqError

log = logging.getLogger("taletrace.learning")

QUIZ_SYSTEM = """Given the text a reader just read aloud and the list of \
words they asked for help with, write 3 short multiple-choice comprehension \
questions. Return ONLY JSON: {"questions": [{"q": "...", "options": \
["...","...","...","..."], "answer_index": 0}, ...]}"""


def end_of_session_quiz(memory_text: str, asked_words: list[str]) -> dict:
    if not memory_text.strip():
        return {"questions": []}
    try:
        raw = call_learning_engine(
            QUIZ_SYSTEM,
            json.dumps({"text": memory_text[-4000:], "asked_words": asked_words}),
            json_mode=True,
        )
        return json.loads(raw)
    except (GroqError, json.JSONDecodeError) as e:
        log.error("Quiz generation failed: %s", e)
        return {"questions": []}
