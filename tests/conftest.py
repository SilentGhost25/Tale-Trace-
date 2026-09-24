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
