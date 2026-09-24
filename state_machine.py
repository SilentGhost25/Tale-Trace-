"""
Maps the ESP32 button combo to one of four modes, per spec:

  momentary=0, toggle=0  -> IDLE_READING   (OCR+merge engine runs, TTS plays)
  momentary=0, toggle=1  -> MEANING_MODE   (gesture pointing, TTS paused)
  momentary=1, toggle=0  -> UPDATE_POSITION(reposition reading pointer, TTS paused then resumes from there)
  momentary=1, toggle=1  -> SCROLL_MODE    (OLED scrolls meaning text; firmware also edge-triggers locally)
"""
from enum import Enum, auto

from esp32_client import ButtonState


class Mode(Enum):
    IDLE_READING = auto()
    MEANING_MODE = auto()
    UPDATE_POSITION = auto()
    SCROLL_MODE = auto()


def determine_mode(state: ButtonState) -> Mode:
    if state.toggle and state.momentary:
        return Mode.SCROLL_MODE
    if state.toggle:
        return Mode.MEANING_MODE
    if state.momentary:
        return Mode.UPDATE_POSITION
    return Mode.IDLE_READING


def tts_should_be_paused(mode: Mode) -> bool:
    # Spec: "when any buttons are pressed the text-to-speech must be paused;
    # when [no buttons are on], it can resume."
    return mode != Mode.IDLE_READING
