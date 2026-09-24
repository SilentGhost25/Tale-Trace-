"""
Offline Text-to-Speech (TTS) Engine for TaleTrace.

Key Requirements:
- 100% Offline (Windows SAPI5 / pyttsx3, zero network dependency).
- Stutter-free sentence playback with native word-boundary tracking ('started-word' event).
- Pauses immediately when any button is pressed / active.
- Resumes seamlessly from the exact word index when both buttons are released.
- Allows jumping reading position (Update Position mode).
"""
import logging
import threading
import time
from typing import Dict, List, Optional

import pyttsx3

from config import settings

log = logging.getLogger("taletrace.tts")


class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._words: List[dict] = []  # [{"w": str, "start": int, "end": int}]
        self._current_index = 0
        self._paused = True
        self._stop_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def load_pointer_map(self, pointer_map: Dict) -> None:
        """Loads updated word pointer map while preserving the reading cursor."""
        with self._lock:
            new_words = pointer_map.get("words", [])
            if not new_words:
                return

            self._words = new_words
            if self._current_index >= len(self._words):
                self._current_index = max(0, len(self._words) - 1)

    def set_position_by_char_offset(self, char_offset: int) -> bool:
        """Jumps the TTS pointer to the word covering this character offset."""
        with self._lock:
            for i, w in enumerate(self._words):
                if w["start"] <= char_offset < w["end"]:
                    self._current_index = i
                    log.info("TTS position set to word #%d: '%s' (offset %d)", i, w["w"], char_offset)
                    return True
            log.warning("Char offset %d not found in pointer map; position unchanged.", char_offset)
            return False

    def set_position_by_word(self, target_word: str) -> bool:
        """Finds closest occurrence of target_word in the pointer map and jumps to it."""
        with self._lock:
            target_clean = target_word.strip().lower()
            # First search forward from current position
            for i in range(self._current_index, len(self._words)):
                if target_clean in self._words[i]["w"].lower():
                    self._current_index = i
                    log.info("TTS position jumped forward to word #%d: '%s'", i, self._words[i]["w"])
                    return True
            # Search backward if not found forward
            for i in range(0, self._current_index):
                if target_clean in self._words[i]["w"].lower():
                    self._current_index = i
                    log.info("TTS position jumped backward to word #%d: '%s'", i, self._words[i]["w"])
                    return True

            log.warning("Word '%s' not found in pointer map.", target_word)
            return False

    def pause(self) -> None:
        """Instantly halts playback and sets paused state."""
        self._stop_flag.set()
        self._paused = True
        log.debug("TTS paused at word #%d", self._current_index)

    def resume(self) -> None:
        """Resumes playback from the exact paused word index."""
        if not self._paused:
            return

        with self._lock:
            if not self._words or self._current_index >= len(self._words):
                log.debug("No text to speak or at end of text.")
                return

        self._paused = False
        self._stop_flag.clear()

        # If previous worker thread is still running, wait briefly
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.3)

        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        log.info("TTS resumed from word #%d: '%s'", self._current_index, self._words[self._current_index]["w"])

    def _speech_worker(self) -> None:
        """
        Background speech thread with SAPI5 COM initialization.
        Speaks sentence chunks naturally from merge memory while keeping
        the reading index accurate and continuous.
        """
        engine = None
        try:
            # Initialize COM and pyttsx3 on this dedicated thread
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

            engine = pyttsx3.init()
            engine.setProperty("rate", 160)  # Moderate comfortable storytelling pace

            words_spoken_in_chunk = 0
            current_chunk_start = 0
            current_chunk_len = 0

            def on_word_event(name, location, length):
                nonlocal words_spoken_in_chunk
                if self._stop_flag.is_set():
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    return

                words_spoken_in_chunk += 1
                with self._lock:
                    target_idx = current_chunk_start + min(words_spoken_in_chunk, current_chunk_len)
                    if target_idx < len(self._words):
                        self._current_index = target_idx

            engine.connect("started-word", on_word_event)

            idle_wait_count = 0
            while not self._stop_flag.is_set():
                with self._lock:
                    if self._current_index >= len(self._words):
                        # At the end of current memory; wait briefly for new OCR frames to append words
                        chunk_words = []
                    else:
                        start_idx = self._current_index
                        current_chunk_start = start_idx
                        chunk_words = []
                        for idx in range(start_idx, min(start_idx + 10, len(self._words))):
                            w_item = self._words[idx]
                            chunk_words.append(w_item["w"])
                            # Stop at sentence boundaries for natural cadence
                            if w_item["w"] and w_item["w"][-1] in ".!?":
                                break

                if not chunk_words:
                    # Wait for new words to arrive from camera/OCR
                    if self._stop_flag.wait(timeout=0.6):
                        break
                    idle_wait_count += 1
                    # After 10 idle checks (~6 seconds with no new text), park worker
                    if idle_wait_count >= 10:
                        break
                    continue

                idle_wait_count = 0
                current_chunk_len = len(chunk_words)
                words_spoken_in_chunk = 0
                chunk_text = " ".join(chunk_words)

                try:
                    # Clean up pyttsx3 internal loop state if previously interrupted
                    if hasattr(engine, "_inLoop") and engine._inLoop:
                        engine._inLoop = False
                    if hasattr(engine, "proxy") and hasattr(engine.proxy, "_inLoop") and engine.proxy._inLoop:
                        engine.proxy._inLoop = False

                    engine.say(chunk_text)
                    engine.runAndWait()
                except RuntimeError as re:
                    log.warning("pyttsx3 loop busy or interrupted (%s); resetting engine.", re)
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 160)
                    engine.connect("started-word", on_word_event)
                except Exception as ex:
                    log.error("TTS chunk playback error: %s", ex)

                if not self._stop_flag.is_set():
                    with self._lock:
                        self._current_index = current_chunk_start + current_chunk_len

        except Exception as e:
            log.error("TTS speech worker error: %s", e)
        finally:
            if engine:
                try:
                    if hasattr(engine, "_inLoop") and engine._inLoop:
                        engine._inLoop = False
                    engine.stop()
                except Exception:
                    pass
            self._paused = True
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except (ImportError, Exception):
                pass

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_word_index(self) -> int:
        with self._lock:
            return self._current_index

    @property
    def current_word(self) -> str:
        with self._lock:
            if self._words and 0 <= self._current_index < len(self._words):
                return self._words[self._current_index]["w"]
            return ""


tts = TTSEngine()
