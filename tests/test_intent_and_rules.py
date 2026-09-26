import pytest
from unittest.mock import MagicMock
from esp32_client import ButtonState
import ai_engine
import merge_engine
from tts_engine import TTSEngine


def test_button_state_intent_fields():
    # Default intent fields
    s = ButtonState(momentary=False, toggle=False)
    assert s.intend is False
    assert s.intend_pressed is False
    assert s.intend_released is False

    # Custom intent fields
    s2 = ButtonState(momentary=False, toggle=True, intend=True, intend_pressed=True, intend_released=False)
    assert s2.intend is True
    assert s2.intend_pressed is True
    assert s2.intend_released is False


def test_meaning_of_normalization_and_caching(monkeypatch):
    ai_engine._meaning_cache.clear()

    mock_explain = MagicMock(return_value="A creature with seven arms.")
    monkeypatch.setattr(ai_engine, "explain_word", mock_explain)

    # First lookup
    m1 = ai_engine.meaning_of("Rot8!", "Rot8 was swimming.")
    assert m1 == "a creature with seven arms."
    assert mock_explain.call_count == 1

    # Second lookup with different punctuation/casing should hit cache
    m2 = ai_engine.meaning_of("rot8", "different context")
    assert m2 == "a creature with seven arms."
    assert mock_explain.call_count == 1  # Cache hit, no redundant call!


def test_recap_up_to_no_spoilers(monkeypatch):
    mock_recap = MagicMock(return_value="Rot8 and Tumboo were eating food.")
    monkeypatch.setattr(ai_engine, "generate_recap", mock_recap)

    # Set up test memory
    text = "Rot8 went to the sea. Tumboo stayed on shore. The villain arrived later."
    merge_engine.update_memory_and_pointer_map(text, current_speaking_word_index=4)

    # Recap up to word index 4 ("Rot8 went to the sea.")
    res = merge_engine.recap_up_to(word_index=4, max_lines=4, include_prior_sessions=False)
    assert res == "Rot8 and Tumboo were eating food."

    prompt_arg = mock_recap.call_args[0][0]
    # Ensure future text ("The villain arrived later.") was NOT included in the recap prompt
    assert "The villain arrived" not in prompt_arg
    assert "Rot8 went to the sea" in prompt_arg


def test_tts_engine_methods():
    engine = TTSEngine()
    assert hasattr(engine, "stop_immediately")
    assert hasattr(engine, "speak")
    assert hasattr(engine, "is_speaking")
    assert engine.is_speaking() is False


def test_explain_with_context(monkeypatch):
    mock_call_ai = MagicMock(return_value="Breath means air in lungs.\nIrrit8 says this because Po8 is speaking too much.")
    monkeypatch.setattr(ai_engine, "call_ai_engine", mock_call_ai)

    text = "Rot8 went to the sea. Tumboo stayed on shore."
    merge_engine.update_memory_and_pointer_map(text, current_speaking_word_index=2)
    mem_slice = merge_engine.get_memory_text_up_to(2, include_prior_sessions=False)
    assert "Rot8 went to" in mem_slice
    assert "Tumboo" not in mem_slice

    res = ai_engine.explain_with_context(
        word="breath",
        current_line="Save your breath!",
        memory_text=mem_slice,
    )
    assert "Breath means air" in res
    assert mock_call_ai.called


def test_explain_intent_context(monkeypatch):
    mock_call_ai = MagicMock(return_value="Rot8 is warning Tumboo about the dark fin.\nSnork means a breath underwater.")
    monkeypatch.setattr(ai_engine, "call_ai_engine", mock_call_ai)

    text = "Rot8 went to the sea. Tumboo stayed on shore."
    merge_engine.update_memory_and_pointer_map(text, current_speaking_word_index=2)
    mem_slice = merge_engine.get_memory_text_up_to(2, include_prior_sessions=False)

    res = ai_engine.explain_intent_context(
        word="snork",
        current_line="Take a snork now!",
        memory_text=mem_slice,
    )
    assert "Rot8 is warning" in res
    assert mock_call_ai.called
