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


def _get_candidate_keys(primary_key: str) -> list[str]:
    """Returns candidate API keys starting with the primary subsystem key."""
    candidates = []
    if primary_key:
        candidates.append(primary_key)
    for k in [settings.groq_api_key, settings.groq_key_ai, settings.groq_key_merge, settings.groq_key_learning]:
        if k and k not in candidates:
            candidates.append(k)
    return candidates


def _chat(api_key: str, model: str, system: str, user: str,
          json_mode: bool = False, temperature: float = 0.2) -> str:
    candidate_keys = _get_candidate_keys(api_key)
    if not candidate_keys:
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

    last_err = None
    for attempt, key in enumerate(candidate_keys):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                data=json.dumps(body),
                timeout=15,
            )
            if resp.status_code == 429:
                log.warning("Groq key #%d hit rate limit (429). Rotating to next available key...", attempt + 1)
                last_err = GroqError(f"Groq 429 Rate Limit: {resp.text}")
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            last_err = e
            log.warning("Groq request failed on key #%d: %s. Rotating to next key...", attempt + 1, e)

    raise GroqError(f"All available Groq API keys exhausted: {last_err}")


def call_merge_engine(system: str, user: str, json_mode: bool = False,
                       fast: bool = False) -> str:
    model = settings.groq_fast_model if fast else settings.groq_model
    return _chat(settings.groq_key_merge, model, system, user, json_mode=json_mode)


def call_ai_engine(system: str, user: str, json_mode: bool = False) -> str:
    return _chat(settings.groq_key_ai, settings.groq_model, system, user, json_mode=json_mode)


def call_learning_engine(system: str, user: str, json_mode: bool = False) -> str:
    return _chat(settings.groq_key_learning, settings.groq_model, system, user, json_mode=json_mode)
