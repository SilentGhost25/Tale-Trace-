"""
Thin Groq (OpenAI-compatible) chat client. Three call sites build their own
client from their own key (merge / ai / learning) per the .env design note:
the merge engine is called several times a second while the camera runs, the
AI engine once per reader request — sharing a key would let the camera loop
exhaust the rate limit and have the reader see the failure.
"""
import json
import logging
import requests
from typing import Optional

from config import settings

log = logging.getLogger("taletrace.groq")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(Exception):
    pass


def _chat(api_key: str, model: str, system: str, user: str,
          json_mode: bool = False, temperature: float = 0.2) -> str:
    if not api_key:
        raise GroqError("No Groq API key configured for this subsystem.")

    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise GroqError(f"Groq request failed: {e}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise GroqError(f"Unexpected Groq response shape: {data}") from e


def call_merge_engine(system: str, user: str, json_mode: bool = False,
                       fast: bool = False) -> str:
    model = settings.groq_fast_model if fast else settings.groq_model
    return _chat(settings.groq_key_merge, model, system, user, json_mode=json_mode)


def call_ai_engine(system: str, user: str, json_mode: bool = False) -> str:
    return _chat(settings.groq_key_ai, settings.groq_model, system, user, json_mode=json_mode)


def call_learning_engine(system: str, user: str, json_mode: bool = False) -> str:
    return _chat(settings.groq_key_learning, settings.groq_model, system, user, json_mode=json_mode)
