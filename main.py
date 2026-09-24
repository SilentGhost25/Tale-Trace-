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
from merge_engine import clean_ocr_text, update_memory_and_pointer_map, get_memory_text
from ai_engine import explain_word
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

# ── Background worker ────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taletrace-bg")
_pending: Optional[Future] = None


def _task_busy() -> bool:
    """True if a background task is still running."""
    return _pending is not None and not _pending.done()


# ── Background-safe heavy work functions ─────────────────────────────────

def _bg_idle_ocr(webcam: WebcamStream) -> None:
    """Background: OCR → clean → merge memory → update TTS pointer map."""
    global _last_captured_ocr_text
    try:
        frame = webcam.get_preprocessed_frame()
        if frame is None:
            return
        ocr_result = ocr_image(frame)
        raw_text = ocr_result.text.strip()
        if raw_text and raw_text != _last_captured_ocr_text:
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
    """Background: finger check (local, fast) → OCR → word match → Groq meaning → OLED."""
    global _last_meaning_word, _last_meaning_time
    try:
        now = time.time()
        if now - _last_meaning_time < MEANING_DEBOUNCE:
            return

        frame = webcam.get_latest_frame()
        if frame is None:
            return

        # ── Fast local finger check FIRST (~50 ms, no network) ──
        if not quick_finger_check(frame):
            return  # No finger visible → skip expensive OCR call entirely

        # Finger detected → now do OCR + word selection
        ocr_result = ocr_image(frame)
        if not ocr_result.words:
            return

        pointed = find_pointed_word(frame, ocr_result)
        if pointed is None or not pointed.word:
            return

        # Avoid repeatedly fetching explanation if finger is hovering over same word
        if pointed.word == _last_meaning_word:
            return

        _last_meaning_word = pointed.word
        _last_meaning_time = now
        log.info("Meaning mode: Pointed word '%s' in context: '%s'", pointed.word, pointed.context)

        meaning = explain_word(pointed.word, pointed.context)

        # If unable to display the definition, do not display anything on OLED
        if not meaning or not meaning.strip():
            log.warning("No definition available for '%s'; not displaying anything.", pointed.word)
            return

        _asked_words.append(pointed.word)
        display_msg = f"{pointed.word}: {meaning}"
        send_display_text(display_msg[:DISPLAY_CHAR_LIMIT])
    except Exception as e:
        log.error("Background meaning lookup error: %s", e)


def _bg_update_position(webcam: WebcamStream) -> None:
    """Background: finger check (local, fast) → OCR → word match → jump TTS position."""
    try:
        frame = webcam.get_latest_frame()
        if frame is None:
            return

        # ── Fast local finger check FIRST (~50 ms, no network) ──
        if not quick_finger_check(frame):
            return  # No finger visible → skip expensive OCR call entirely

        ocr_result = ocr_image(frame)
        if not ocr_result.words:
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


# ── Main loop ────────────────────────────────────────────────────────────

def main() -> None:
    global _pending

    print("=" * 65)
    print("📚 TaleTrace: Smart Book-Reading Companion (2026 Edition)")
    print(f"📖 Book: {settings.book_title} by {settings.book_author}")
    print("🎥 Camera: Logitech C270 HD WebCam (1280x720 DirectShow)")
    print("🔘 Button 1 (Momentary): Update Reading Position")
    print("🔘 Button 2 (Toggle)   : Meaning Mode (Fingertip Explanation)")
    print("🔘 Both Buttons ON     : Scroll Meaning Text on OLED")
    print("=" * 65 + "\n")

    webcam = WebcamStream()
    try:
        webcam.start()
    except Exception as e:
        log.error("Webcam initialization failed: %s", e)
        print(f"⚠️ Could not start Logitech C270 camera: {e}")
        return

    last_ocr_time = 0.0
    last_mode: Mode = Mode.IDLE_READING

    # Software latch for Meaning Mode: stays ON until toggled again
    _meaning_latched = False
    _prev_toggle_raw = False  # tracks previous raw toggle pin to detect rising edges

    log.info("TaleTrace started. Running master control loop. Press Ctrl+C to stop.")
    try:
        while True:
            button_state = get_button_state()

            # --- Detect rising edge on the hardware toggle button ---
            toggle_now = button_state.toggle
            toggle_rising_edge = toggle_now and not _prev_toggle_raw
            _prev_toggle_raw = toggle_now

            if toggle_rising_edge:
                _meaning_latched = not _meaning_latched
                log.info("Meaning mode latch %s", "ON" if _meaning_latched else "OFF")

            # --- Determine effective mode (instant — no blocking) ---
            if _meaning_latched:
                if button_state.momentary:
                    mode = Mode.SCROLL_MODE
                else:
                    mode = Mode.MEANING_MODE
            else:
                mode = determine_mode(button_state)

            # --- Handle mode transitions immediately ---
            if mode != last_mode:
                log.info("Mode transition: %s -> %s", last_mode.name, mode.name)
                if tts_should_be_paused(mode):
                    tts.pause()
                elif mode == Mode.IDLE_READING and tts.is_paused:
                    tts.resume()

            # --- Submit work to background thread (non-blocking) ---
            if mode == Mode.IDLE_READING:
                # Resume TTS every iteration (in case it paused itself at end of words)
                if tts.is_paused:
                    tts.resume()
                # Submit OCR pipeline only when interval elapsed & no task running
                now = time.time()
                if now - last_ocr_time >= OCR_INTERVAL and not _task_busy():
                    if webcam.is_frame_stable():
                        _pending = _executor.submit(_bg_idle_ocr, webcam)
                    last_ocr_time = now

            elif mode == Mode.MEANING_MODE:
                tts.pause()
                if not _task_busy():
                    _pending = _executor.submit(_bg_meaning_lookup, webcam)

            elif mode == Mode.UPDATE_POSITION:
                tts.pause()
                if not _task_busy():
                    _pending = _executor.submit(_bg_update_position, webcam)

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

