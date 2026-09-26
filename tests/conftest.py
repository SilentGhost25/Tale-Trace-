import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Alias gesture package as gesture_engine for compatibility with ported tests
import gesture
import gesture.models
import gesture.word_selector
import gesture.finger_detector

sys.modules["gesture_engine"] = gesture
sys.modules["gesture_engine.models"] = gesture.models
sys.modules["gesture_engine.word_selector"] = gesture.word_selector
sys.modules["gesture_engine.finger_detector"] = gesture.finger_detector

import pytest

@pytest.fixture(autouse=True)
def reset_memory_state():
    try:
        from ocr_memory import pipeline
        pipeline.merge_memory = ""
        pipeline.permanent_memory = []
        pipeline.current_reading_pointer = 0
        pipeline.current_word_index = 0
    except Exception:
        pass
    yield
    try:
        from ocr_memory import pipeline
        pipeline.merge_memory = ""
        pipeline.permanent_memory = []
        pipeline.current_reading_pointer = 0
        pipeline.current_word_index = 0
    except Exception:
        pass

