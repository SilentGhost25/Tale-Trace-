"""
Edge-TTS Text-to-Speech (TTS) Engine for TaleTrace.

Key Requirements:
- High-fidelity human-sounding neural male voice (Microsoft Edge TTS: en-US-GuyNeural).
- Real-time word-boundary tracking for pointer synchronisation.
- Pauses immediately when any button is pressed / active.
- Resumes seamlessly from the exact word index when both buttons are released.
- Allows jumping reading position (Update Position mode).
- Non-blocking custom announcements for word meanings and story recaps with instant tap-to-stop.
"""
import asyncio
import ctypes
import logging
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional

import edge_tts

from config import settings

log = logging.getLogger("taletrace.tts")

# High-fidelity natural human male voice
DEFAULT_MALE_VOICE = "en-US-GuyNeural"


class MCIAudioPlayer:
    """Windows MCI audio player for low-latency MP3 playback with zero external dependencies."""

    def __init__(self, alias: str):
        self.alias = alias
        self.winmm = ctypes.windll.winmm
        self._buf = ctypes.create_unicode_buffer(256)
        self._lock = threading.Lock()
        self._open = False

    def _send(self, cmd: str) -> str:
        with self._lock:
            self.winmm.mciSendStringW(cmd, self._buf, 255, None)
            return self._buf.value

    def load_and_play(self, file_path: str) -> bool:
        self.close()
        abs_path = os.path.abspath(file_path)
        err = self._send(f'open "{abs_path}" type mpegvideo alias {self.alias}')
        self._open = True
        self._send(f"play {self.alias}")
        return True

    def pause(self) -> None:
        if self._open:
            self._send(f"pause {self.alias}")

    def resume(self) -> None:
        if self._open:
            self._send(f"resume {self.alias}")

    def stop(self) -> None:
        if self._open:
            self._send(f"stop {self.alias}")

    def close(self) -> None:
        if self._open:
            self._send(f"stop {self.alias}")
            self._send(f"close {self.alias}")
            self._open = False

    def get_status(self) -> str:
        if not self._open:
            return "stopped"
        return self._send(f"status {self.alias} mode").strip().lower()

    def is_playing(self) -> bool:
        return self.get_status() == "playing"

    def get_position_ms(self) -> int:
        if not self._open:
            return 0
        val = self._send(f"status {self.alias} position")
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0


class TTSEngine:
    """
    Edge-TTS powered speech engine with word-level pointer tracking
    and instant hardware button pause/resume/stop controls.
    """

    def __init__(self, voice: Optional[str] = None):
        self._lock = threading.Lock()
        self._words: List[dict] = []  # [{"w": str, "start": int, "end": int}]
        self._current_index = 0
        self._paused = True

        cfg_voice = getattr(settings, "edge_tts_voice", "")
        if voice:
            self.voice = voice
        elif cfg_voice and "Aria" not in cfg_voice:
            self.voice = cfg_voice
        else:
            self.voice = DEFAULT_MALE_VOICE

        self._stop_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._custom_stop_flag = threading.Event()
        self._custom_worker_thread: Optional[threading.Thread] = None
        self._is_custom_speaking = False

        inst_id = id(self)
        self._reading_player = MCIAudioPlayer(f"ttsr_{inst_id}")
        self._custom_player = MCIAudioPlayer(f"ttsc_{inst_id}")

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
            # Search forward from current position
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
        self._reading_player.stop()
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

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.3)

        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        log.info("TTS resumed from word #%d: '%s'", self._current_index, self._words[self._current_index]["w"])

    def _speech_worker(self) -> None:
        """
        Background speech thread using edge-tts.
        Speaks sentence chunks naturally from merge memory while keeping
        the reading index accurate and continuous.
        """
        idle_wait_count = 0
        try:
            while not self._stop_flag.is_set():
                with self._lock:
                    if self._current_index >= len(self._words):
                        chunk_words = []
                    else:
                        start_idx = self._current_index
                        current_chunk_start = start_idx
                        chunk_words = []
                        for idx in range(start_idx, min(start_idx + 12, len(self._words))):
                            w_item = self._words[idx]
                            chunk_words.append(w_item["w"])
                            if w_item["w"] and w_item["w"][-1] in ".!?":
                                break

                if not chunk_words:
                    if self._stop_flag.wait(timeout=0.6):
                        break
                    idle_wait_count += 1
                    if idle_wait_count >= 10:
                        break
                    continue

                idle_wait_count = 0
                current_chunk_len = len(chunk_words)
                chunk_text = " ".join(chunk_words)

                # Synthesize chunk via edge-tts with word boundaries
                try:
                    audio_bytes, boundaries = self._synthesize_sync(chunk_text, word_boundary=True)
                except Exception as ex:
                    log.error("Edge-TTS synthesis error: %s", ex)
                    if self._stop_flag.wait(timeout=1.0):
                        break
                    continue

                if self._stop_flag.is_set() or not audio_bytes:
                    break

                # Play via MCI audio player
                tmp_fd, tmp_file = tempfile.mkstemp(suffix=".mp3", prefix="taletrace_reading_")
                os.close(tmp_fd)
                try:
                    with open(tmp_file, "wb") as f:
                        f.write(audio_bytes)

                    self._reading_player.load_and_play(tmp_file)

                    # Monitor playback and update word pointer in real time
                    while not self._stop_flag.is_set() and self._reading_player.is_playing():
                        pos_ms = self._reading_player.get_position_ms()
                        for b_idx, b in enumerate(boundaries):
                            if pos_ms >= b["offset_ms"]:
                                target_idx = current_chunk_start + min(b_idx, current_chunk_len - 1)
                                with self._lock:
                                    if target_idx < len(self._words):
                                        self._current_index = target_idx
                        time.sleep(0.04)

                    # If chunk played to completion, advance cursor past chunk
                    if not self._stop_flag.is_set():
                        with self._lock:
                            self._current_index = current_chunk_start + current_chunk_len
                finally:
                    self._reading_player.close()
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except Exception:
                            pass

        except Exception as e:
            log.error("TTS speech worker error: %s", e)
        finally:
            self._reading_player.close()
            self._paused = True

    def stop_immediately(self) -> None:
        """Instantly halts any active speech (continuous reading or custom announcement)."""
        self._stop_flag.set()
        self._custom_stop_flag.set()
        self._paused = True
        self._is_custom_speaking = False
        self._reading_player.close()
        self._custom_player.close()
        log.info("TTS stopped immediately.")

    def speak(self, text: str) -> None:
        """Speaks arbitrary announcement text (e.g. word meaning or session recap) non-blocking."""
        if not text or not text.strip():
            return

        self.stop_immediately()

        if self._custom_worker_thread and self._custom_worker_thread.is_alive():
            self._custom_worker_thread.join(timeout=0.2)

        self._custom_stop_flag.clear()
        self._custom_worker_thread = threading.Thread(
            target=self._custom_speech_worker,
            args=(text.strip(),),
            daemon=True,
        )
        self._custom_worker_thread.start()

    def _custom_speech_worker(self, text: str) -> None:
        """Speaks one-shot announcements (word explanation / recap) via edge-tts."""
        self._is_custom_speaking = True
        tmp_file = ""
        try:
            audio_bytes, _ = self._synthesize_sync(text, word_boundary=False)
            if self._custom_stop_flag.is_set() or not audio_bytes:
                return

            tmp_fd, tmp_file = tempfile.mkstemp(suffix=".mp3", prefix="taletrace_custom_")
            os.close(tmp_fd)

            with open(tmp_file, "wb") as f:
                f.write(audio_bytes)

            self._custom_player.load_and_play(tmp_file)

            while not self._custom_stop_flag.is_set() and self._custom_player.is_playing():
                time.sleep(0.04)

        except Exception as e:
            log.error("TTS custom announcement error: %s", e)
        finally:
            self._custom_player.close()
            self._is_custom_speaking = False
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def _synthesize_sync(self, text: str, word_boundary: bool = False) -> tuple[bytes, list]:
        """Synchronously calls edge-tts communicate with retry for network resilience."""
        async def _run():
            boundary_arg = "WordBoundary" if word_boundary else "SentenceBoundary"
            comm = edge_tts.Communicate(text, self.voice, boundary=boundary_arg)
            audio = bytearray()
            boundaries = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    boundaries.append({
                        "text": chunk.get("text", ""),
                        "offset_ms": chunk.get("offset", 0) / 10000,
                        "duration_ms": chunk.get("duration", 0) / 10000,
                    })
            return bytes(audio), boundaries

        # Retry up to 3 times on transient socket / network resets
        last_err = None
        for attempt in range(3):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_run())
            except Exception as e:
                last_err = e
                log.warning("Edge-TTS connection retry %d/3: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.3 * (attempt + 1))
            finally:
                loop.close()

        # Offline fallback: if Bing endpoint drops, synthesize cleanly via local Windows SAPI5
        log.warning("Edge-TTS endpoint unavailable (%s). Falling back to offline local voice.", last_err)
        try:
            return self._synthesize_sapi_fallback(text)
        except Exception as fallback_err:
            log.error("Offline speech fallback also failed: %s", fallback_err)

        return bytes(), []

    def _synthesize_sapi_fallback(self, text: str) -> tuple[bytes, list]:
        """Offline fallback using Windows SAPI5 to ensure speech is NEVER lost."""
        import pyttsx3
        tmp_wav = ""
        try:
            tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="taletrace_sapi_")
            os.close(tmp_fd)

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            # Select David Desktop (Male) if available
            for v in voices:
                if "david" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 165)
            engine.save_to_file(text, tmp_wav)
            engine.runAndWait()

            with open(tmp_wav, "rb") as f:
                data = f.read()

            words = text.split()
            boundaries = []
            approx_duration_per_word = 320  # ms
            for idx, w in enumerate(words):
                boundaries.append({
                    "text": w,
                    "offset_ms": idx * approx_duration_per_word,
                    "duration_ms": approx_duration_per_word,
                })
            return data, boundaries
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def is_speaking(self) -> bool:
        """Returns True if book reading or custom announcement is currently playing."""
        if self._is_custom_speaking:
            return True
        return not self._paused and self._worker_thread is not None and self._worker_thread.is_alive()

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
