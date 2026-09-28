"""
TaleTrace Master Controller.

Orchestrates:
- Logitech C270 HD WebCam 720p capture stream.
- Hardware button state polling (ESP32 DevKit via Serial UART / HTTP).
- 4 Operational Modes:
  1. IDLE_READING   (Both OFF): OCR -> Groq Merge -> Python State Code -> Offline TTS.
  2. MEANING_MODE   (Toggle ON, Momentary OFF): Gesture pointing -> Groq meaning -> OLED display (TTS paused).
  3. SCROLL_MODE    (Toggle ON, Momentary ON): OLED scrolls meaning text (TTS paused).
  4. UPDATE_POSITION(Toggle OFF, Momentary ON): Gesture pointing -> TTS pointer jumps to word (TTS paused until release).

All heavy work (OCR, Groq API calls, gesture detection) runs in a background
thread so the main button-polling loop stays responsive (<200 ms per iteration).
"""
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

# Ensure UTF-8 console output on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from config import settings
from webcam_capture import WebcamStream
from esp32_client import get_button_state, send_display_text
from state_machine import determine_mode, tts_should_be_paused, Mode
from ocr_engine import ocr_image
from merge_engine import clean_ocr_text, update_memory_and_pointer_map, get_memory_text, recap_up_to, get_memory_text_up_to, get_pointer_map
from ai_engine import explain_word, explain_with_context, meaning_of, explain_intent_context
from gesture_engine import find_pointed_word, quick_finger_check
from tts_engine import tts
from learning_engine import end_of_session_quiz

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("taletrace.main")

POLL_INTERVAL = 0.10     # Button polling interval (seconds) — kept fast for instant toggle
OCR_INTERVAL = settings.ocr_capture_interval  # OCR capture duration (defaults to 6.0s to avoid API rate limits)
DISPLAY_CHAR_LIMIT = 400 # Maximum characters sent to OLED display
MEANING_DEBOUNCE = 3.0   # Cooldown between meaning requests (seconds)

_asked_words: list[str] = []
_last_meaning_word: str = ""
_last_meaning_time: float = 0.0
_last_captured_ocr_text: str = ""
_last_looked_up_meaning: str = ""
_tts_trigger: Optional[str] = None       # None | "INTENT" | "MEANING"
_intent_recap_active: bool = False      # Tap-to-start / Tap-to-stop toggle state (INTENT_MODE)
_last_intent_task_ms: int = 0
_last_intent_word: Optional[str] = None
INTENT_TASK_MIN_INTERVAL_MS: int = 1200  # 1.2s between fingertip lookups
_last_meaning_task_time: float = 0.0    # Throttle: min interval between MEANING_MODE task submissions
_last_update_task_time: float = 0.0     # Throttle: min interval between UPDATE_POSITION task submissions
MODE_TASK_MIN_INTERVAL: float = 1.0     # Minimum seconds between task submissions per mode

# ── Background worker ────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taletrace-bg")
_pending: Optional[Future] = None


def _task_busy() -> bool:
    """True if a background task is still running. Logs exceptions if finished."""
    global _pending
    if _pending is None:
        return False
    if _pending.done():
        exc = _pending.exception()
        if exc is not None:
            log.error("Background task raised uncaught exception: %s", exc)
        _pending = None
        return False
    return True


# ── Background-safe heavy work functions ─────────────────────────────────

def _bg_idle_ocr(webcam: WebcamStream) -> None:
    """Background: OCR → clean → merge memory → update TTS pointer map."""
    global _last_captured_ocr_text
    try:
        frame = webcam.get_latest_frame()
        if frame is None:
            return
        ocr_result = ocr_image(frame)
        raw_text = ocr_result.text.strip()
        if raw_text:
            if _last_captured_ocr_text:
                tokens_old = set(w.strip(".,!?;:\"'()").lower() for w in _last_captured_ocr_text.split() if len(w) > 2)
                tokens_new = set(w.strip(".,!?;:\"'()").lower() for w in raw_text.split() if len(w) > 2)
                if tokens_new and len(tokens_new.intersection(tokens_old)) / len(tokens_new) >= 0.85:
                    return

            _last_captured_ocr_text = raw_text
            cleaned = clean_ocr_text(raw_text)
            if cleaned:
                current_word_idx = tts.current_word_index
                pointer_map = update_memory_and_pointer_map(
                    cleaned,
                    current_speaking_word_index=current_word_idx,
                )
                tts.load_pointer_map(pointer_map)
    except Exception as e:
        log.error("Background OCR pipeline error: %s", e)


def _bg_meaning_lookup(webcam: WebcamStream) -> None:
    """Background: OCR → word match → Groq meaning → OLED + conditional TTS (Rule 1)."""
    global _last_meaning_word, _last_meaning_time, _tts_trigger, _last_looked_up_meaning
    try:
        now = time.time()
        if now - _last_meaning_time < 1.0:  # Reduced cooldown to 1.0s for responsive feedback
            return

        log.info("[MEANING] Background task fired - capturing frame for finger & word selection...")

        frame = webcam.get_latest_frame()
        if frame is None:
            log.warning("[MEANING] Could not retrieve frame from webcam.")
            return

        ocr_result = ocr_image(frame)
        if not ocr_result or not ocr_result.words:
            log.warning("[MEANING] OCR produced no words from frame.")
            return

        pointed = find_pointed_word(frame, ocr_result)
        if pointed is None or not pointed.word:
            log.info("[MEANING] Could not correlate fingertip to a specific word.")
            return

        # Avoid repeatedly fetching explanation if finger is hovering over same word
        if pointed.word == _last_meaning_word:
            return

        _last_meaning_word = pointed.word
        _last_meaning_time = now
        log.info("Meaning mode: Pointed word '%s' in context: '%s'", pointed.word, pointed.context)

        # Slice memory strictly up to current reading pointer for zero-spoiler contextual reasoning
        memory_slice = get_memory_text_up_to(tts.current_word_index)
        meaning = explain_with_context(
            word=pointed.word,
            current_line=pointed.context,
            memory_text=memory_slice,
        )

        # If unable to display the definition, do not display anything on OLED
        if not meaning or not meaning.strip():
            log.warning("No definition available for '%s'; not displaying anything.", pointed.word)
            return

        _asked_words.append(pointed.word)
        display_msg = f"{pointed.word}:\n{meaning}"
        log.info("[DISPLAY] Sending to OLED: %s", display_msg[:60].replace("\n", " | "))
        send_display_text(display_msg[:DISPLAY_CHAR_LIMIT])

        # ── RULE 1: Check if adjacent words or consecutive lookups share the same meaning ──
        my_norm_meaning = meaning_of(pointed.word, pointed.context)
        words_list = [w.text for w in ocr_result.words]
        pointed_idx = -1
        for idx, w in enumerate(ocr_result.words):
            if w.text == pointed.word:
                pointed_idx = idx
                break

        n_prev = words_list[pointed_idx - 1] if pointed_idx > 0 else None
        n_next = words_list[pointed_idx + 1] if 0 <= pointed_idx < len(words_list) - 1 else None

        same_as_prev = bool(n_prev and meaning_of(n_prev, pointed.context) == my_norm_meaning)
        same_as_next = bool(n_next and meaning_of(n_next, pointed.context) == my_norm_meaning)
        same_as_consecutive = bool(_last_looked_up_meaning and _last_looked_up_meaning == my_norm_meaning)

        if same_as_prev or same_as_next or same_as_consecutive:
            log.info("Rule 1 triggered (shared meaning detected) -> speaking meaning aloud.")
            tts.speak(f"Meaning of the word: {meaning}")
            _tts_trigger = "MEANING"
        else:
            # Different meaning: OLED display only, no speech
            log.info("Different meaning -> OLED display only, TTS silent.")

        _last_looked_up_meaning = my_norm_meaning
    except Exception as e:
        log.error("Background meaning lookup error: %s", e)


def _bg_update_position(webcam: WebcamStream) -> None:
    """Background: OCR → word match → jump TTS position."""
    try:
        frame = webcam.get_latest_frame()
        if frame is None:
            return

        ocr_result = ocr_image(frame)
        if not ocr_result or not ocr_result.words:
            return

        pointed = find_pointed_word(frame, ocr_result)
        if pointed is None or not pointed.word:
            return

        success = tts.set_position_by_word(pointed.word)
        if success:
            log.info("Reading position synced to word: '%s'", pointed.word)
            send_display_text(f"Position: {pointed.word}"[:DISPLAY_CHAR_LIMIT])
    except Exception as e:
        log.error("Background position update error: %s", e)


def _bg_intent_context(webcam: WebcamStream) -> None:
    """Intent mode: find pointed word, explain why it's being said now, using merged memory."""
    global _last_intent_word

    try:
        frame = webcam.get_latest_frame()
        if frame is None:
            return

        ocr_result = ocr_image(frame)
        if not ocr_result or not ocr_result.words:
            return

        pointed = find_pointed_word(frame, ocr_result)
        if pointed is None or not pointed.word:
            return

        if pointed.word == _last_intent_word:
            return
        _last_intent_word = pointed.word

        # Build a memory slice up to the pointer — no spoilers
        memory_text = get_memory_text_up_to(tts.current_word_index)

        msg = explain_intent_context(
            word=pointed.word,
            current_line=getattr(pointed, "context", "") or pointed.word,
            memory_text=memory_text,
        )

        if not msg:
            send_display_text("Keep reading for more context.")
            return

        send_display_text(msg[:DISPLAY_CHAR_LIMIT])
        tts.speak(msg)
        log.info("[INTENT] Spoke context for '%s'", pointed.word)

    except Exception as e:
        log.error("_bg_intent_context failed: %s", e)


# ── Main loop ────────────────────────────────────────────────────────────

def main() -> None:
    global _pending, _tts_trigger, _intent_recap_active
    global _last_intent_task_ms, _last_intent_word
    global _last_meaning_task_time, _last_update_task_time

    print("=" * 65)
    print("📚 TaleTrace: Smart Book-Reading Companion (2026 Edition)")
    print(f"📖 Book: {settings.book_title} by {settings.book_author}")
    print("🎥 Camera: Logitech C270 HD WebCam (1280x720 DirectShow)")
    print("🔘 Button 1 (Momentary): Update Reading Position")
    print("🔘 Button 2 (Toggle)   : Meaning Mode (Fingertip Explanation)")
    print("🔘 Both Buttons ON     : Scroll Meaning Text on OLED")
    print("=" * 65 + "\n")

    # Pre-load reading state / pointer map immediately on boot quietly
    existing_map = get_pointer_map()
    if existing_map and existing_map.get("words"):
        tts.load_pointer_map(existing_map)
        log.info("Pre-loaded %d words from reading memory into TTS engine.", len(existing_map["words"]))
        send_display_text(f"TaleTrace Ready!\n({len(existing_map['words'])} words loaded)")
    else:
        send_display_text("TaleTrace Ready!\nPoint at book")

    tts.pause()  # Ensure quiet startup — no auto-reading or summary playback on boot

    webcam = WebcamStream()
    try:
        webcam.start()
    except Exception as e:
        log.error("Webcam initialization failed: %s", e)
        print(f"⚠️ Could not start Logitech C270 camera: {e}")
        return

    last_ocr_time = 0.0
    last_mode: Mode = Mode.IDLE_READING

    log.info("TaleTrace started. Running master control loop. Press Ctrl+C to stop.")
    try:
        while True:
            button_state = get_button_state()
            mode = determine_mode(button_state)

            # --- RULE 0: Immediate hard-stop when trigger conditions change ---
            if _tts_trigger == "MEANING" and mode != Mode.MEANING_MODE:
                tts.stop_immediately()
                _tts_trigger = None

            # --- INTENT MODE: physical toggle switch (ON / OFF) ---
            intent_active_signal = button_state.intend or button_state.intend_pressed
            if intent_active_signal and not _intent_recap_active:
                tts.stop_immediately()
                tts.pause()
                _intent_recap_active = True
                _tts_trigger = "INTENT"
                _last_intent_word = None
                log.info("Intent mode ON (Toggle ON) — point at a word.")
                send_display_text("Intent mode ON\nPoint at a word...")

            elif not intent_active_signal and _intent_recap_active:
                tts.stop_immediately()
                _intent_recap_active = False
                _tts_trigger = None
                _last_intent_word = None
                log.info("Intent mode OFF (Toggle OFF).")
                send_display_text("Intent mode off.")

            # --- While INTENT MODE is on, run the fingertip → context pipeline ---
            if _intent_recap_active:
                if not _task_busy():
                    now_ms = int(time.time() * 1000)
                    if (now_ms - _last_intent_task_ms) >= INTENT_TASK_MIN_INTERVAL_MS:
                        _pending = _executor.submit(_bg_intent_context, webcam)
                        _last_intent_task_ms = now_ms
                last_mode = mode
                time.sleep(POLL_INTERVAL)
                continue

            # --- Handle mode transitions immediately ---
            if mode != last_mode:
                print(f"\n⚡ MODE → {mode.name}")
                log.info("Mode transition: %s -> %s", last_mode.name, mode.name)
                if mode == Mode.MEANING_MODE:
                    send_display_text("Meaning Mode:\nPoint to a word...")
                elif mode == Mode.UPDATE_POSITION:
                    send_display_text("Position Mode:\nPoint to jump...")
                elif mode == Mode.IDLE_READING and not _intent_recap_active:
                    send_display_text("TaleTrace Ready!\nPoint at book")

                if tts_should_be_paused(mode):
                    tts.pause()
                elif mode == Mode.IDLE_READING and not _intent_recap_active and tts.is_paused:
                    tts.resume()

            # --- Submit work to background thread (non-blocking) ---
            if mode == Mode.IDLE_READING:
                if not _intent_recap_active and _tts_trigger is None and tts.is_paused:
                    tts.resume()
                # Submit OCR pipeline only when interval elapsed & no task running
                now = time.time()
                if now - last_ocr_time >= OCR_INTERVAL and not _task_busy():
                    if webcam.is_frame_stable():
                        _pending = _executor.submit(_bg_idle_ocr, webcam)
                    last_ocr_time = now

            elif mode == Mode.MEANING_MODE:
                tts.pause()
                now = time.time()
                if not _task_busy() and (now - _last_meaning_task_time) >= MODE_TASK_MIN_INTERVAL:
                    _pending = _executor.submit(_bg_meaning_lookup, webcam)
                    _last_meaning_task_time = now

            elif mode == Mode.UPDATE_POSITION:
                tts.pause()
                now = time.time()
                if not _task_busy() and (now - _last_update_task_time) >= MODE_TASK_MIN_INTERVAL:
                    _pending = _executor.submit(_bg_update_position, webcam)
                    _last_update_task_time = now

            elif mode == Mode.SCROLL_MODE:
                tts.pause()

            last_mode = mode
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt received. Stopping TaleTrace session...")
        tts.pause()
        quiz = end_of_session_quiz(get_memory_text(), _asked_words)
        if quiz.get("questions"):
            log.info("End-of-session comprehension quiz generated (%d questions).", len(quiz["questions"]))
            print(f"📝 Quiz generated: {len(quiz['questions'])} questions saved.")
    finally:
        _executor.shutdown(wait=False)
        webcam.stop()
        print("👋 Session finalized successfully.")


if __name__ == "__main__":
    main()

