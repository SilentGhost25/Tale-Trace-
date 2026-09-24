import pytest
from gesture.models import (
    OCRWord,
    FingerPoint,
    SelectionConfig,
    SelectionResult,
    SelectionStatus,
    SelectionStrategy,
)
from gesture.word_selector import (
    group_ocr_words,
    get_box_distance,
    select_intended_word,
)
from gesture import select_word
from state_machine import determine_mode, tts_should_be_paused, Mode
from esp32_client import ButtonState
import merge_engine


def test_ocr_word_model():
    w = OCRWord(
        text="Rot8",
        bbox=(10, 20, 50, 40),
        center_x=30.0,
        center_y=30.0,
        confidence=0.95,
    )
    assert w.text == "Rot8"
    assert w.bbox == (10, 20, 50, 40)
    assert w.center_x == 30.0
    assert w.center_y == 30.0


def test_line_clustering_and_word_grouping():
    words = [
        OCRWord(text="Rot8", bbox=(10, 20, 60, 40), center_x=35.0, center_y=30.0),
        OCRWord(text="swam", bbox=(70, 20, 120, 40), center_x=95.0, center_y=30.0),
        OCRWord(text="Tumboo", bbox=(10, 70, 80, 90), center_x=45.0, center_y=80.0),
        OCRWord(text="waited", bbox=(90, 70, 150, 90), center_x=120.0, center_y=80.0),
    ]
    cfg = SelectionConfig(line_cluster_tolerance=0.5)
    lines = group_ocr_words(words, cfg)
    assert len(lines) == 2
    assert lines[0].words[0].text == "Rot8"
    assert lines[0].words[1].text == "swam"
    assert lines[1].words[0].text == "Tumboo"
    assert lines[1].words[1].text == "waited"


def test_direct_touch_word_selection():
    words = [
        OCRWord(text="Oct-estra", bbox=(100, 100, 200, 130), center_x=150.0, center_y=115.0, confidence=1.0),
        OCRWord(text="concert", bbox=(210, 100, 280, 130), center_x=245.0, center_y=115.0, confidence=1.0),
    ]
    # Finger touch point directly on "Oct-estra"
    finger = FingerPoint(x=150.0, y=115.0, confidence=1.0, direction=(0.0, -1.0))
    cfg = SelectionConfig(selection_strategy=SelectionStrategy.AUTO)
    res = select_intended_word(finger, words, cfg)

    assert res.status == SelectionStatus.SUCCESS
    assert res.selected_word == "Oct-estra"
    assert res.selected_word_bbox == (100, 100, 200, 130)


def test_directional_cone_selection():
    words = [
        OCRWord(text="Zubair", bbox=(100, 50, 180, 80), center_x=140.0, center_y=65.0, confidence=1.0),
        OCRWord(text="Midha", bbox=(190, 50, 260, 80), center_x=225.0, center_y=65.0, confidence=1.0),
    ]
    # Finger point below "Zubair", pointing straight up (0, -1)
    finger = FingerPoint(x=140.0, y=110.0, confidence=1.0, direction=(0.0, -1.0), detection_method="mediapipe")
    cfg = SelectionConfig(selection_strategy=SelectionStrategy.AUTO)
    res = select_intended_word(finger, words, cfg)

    assert res.selected_word == "Zubair"
    assert res.selected_line == "Zubair Midha"


def test_state_machine_modes():
    # 1. Both OFF -> IDLE_READING (TTS plays)
    s1 = ButtonState(momentary=False, toggle=False)
    assert determine_mode(s1) == Mode.IDLE_READING
    assert not tts_should_be_paused(Mode.IDLE_READING)

    # 2. Toggle ON, Momentary OFF -> MEANING_MODE (TTS paused)
    s2 = ButtonState(momentary=False, toggle=True)
    assert determine_mode(s2) == Mode.MEANING_MODE
    assert tts_should_be_paused(Mode.MEANING_MODE)

    # 3. Both ON -> SCROLL_MODE (TTS paused)
    s3 = ButtonState(momentary=True, toggle=True)
    assert determine_mode(s3) == Mode.SCROLL_MODE
    assert tts_should_be_paused(Mode.SCROLL_MODE)

    # 4. Momentary ON, Toggle OFF -> UPDATE_POSITION (TTS paused)
    s4 = ButtonState(momentary=True, toggle=False)
    assert determine_mode(s4) == Mode.UPDATE_POSITION
    assert tts_should_be_paused(Mode.UPDATE_POSITION)


def test_python_code_generation_and_pointer_stability():
    text = "Rot8 swam through the high seas with Tumboo."
    pmap = merge_engine.update_memory_and_pointer_map(text, current_speaking_word_index=3)

    assert len(pmap["words"]) == 8
    assert pmap["words"][0]["w"] == "Rot8"
    assert pmap["words"][7]["w"] == "Tumboo."

    # Verify generated reading_state.py exists and can be imported
    import reading_state
    assert reading_state.BOOK_TITLE == "Septopus: Trouble on the High Cs"
    assert reading_state.CURRENT_WORD_INDEX == 3
    assert reading_state.get_word_at_offset(0) == 0
    assert reading_state.get_next_word_index(3) == 4


def test_meaning_mode_empty_on_failure(monkeypatch):
    import ai_engine
    # When no word is passed
    assert ai_engine.explain_word("", "some context") == ""
    # When Groq fails
    def mock_call_fail(*args, **kwargs):
        raise Exception("API Rate Limit / Network error")
    monkeypatch.setattr(ai_engine, "call_ai_engine", mock_call_fail)
    monkeypatch.setattr(ai_engine, "call_merge_engine", mock_call_fail)
    assert ai_engine.explain_word("Rot8", "Rot8 swam in the sea") == ""


def test_merge_memory_display(capsys):
    import merge_engine
    merge_engine.display_merge_memory_terminal("Rot8 and Tumboo swam together.", current_word_idx=2, total_words=5)
    captured = capsys.readouterr()
    assert "[ACTIVE MERGE MEMORY CONTENT]" in captured.out
    assert "Rot8 and Tumboo swam together." in captured.out
    assert "Active Reading Pointer: Word #2" in captured.out

