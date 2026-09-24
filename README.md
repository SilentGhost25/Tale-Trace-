# TaleTrace

Reads a physical book aloud via webcam + OCR, lets you point at a word to get
its meaning, and lets you correct your reading position — all driven by two
buttons on an ESP32 + OLED rig.

## ⚠️ Security first
The original request pasted a **live WiFi password and three live Groq API
keys** in plaintext. Treat all of those as compromised: **rotate them now**
(Groq console → regenerate each key; your router → change the WiFi
password). Nothing in this project uses those literal values — `.env.example`
has placeholders, and `firmware/buttons_and_oled.ino` has placeholder WiFi
fields. Copy `.env.example` to `.env` and fill in your *new* keys; `.gitignore`
already excludes `.env` from version control.

## Architecture
```
webcam (Logitech C270) --> OCR (OCR.space Engine 2, pytesseract fallback)
                                |
                    +-----------+-----------+
                    |                       |
              Merge Engine             (per-mode dispatch)
        (grammar cleanup, header                |
         stripping, memory,        +------------+------------+------------+
         pointer map)              |            |            |            |
                              IDLE_READING  MEANING_MODE  UPDATE_POSITION SCROLL_MODE
                              (TTS plays)  (gesture -->   (gesture -->    (OLED scrolls,
                                            AI Engine -->  jump TTS        TTS paused;
                                            OLED shows     pointer,        firmware
                                            meaning)       TTS paused)     edge-triggers
                                                                           locally)
```

Mode is decided purely from the ESP32's two buttons, polled over HTTP:

| momentary | toggle | mode              | what happens |
|-----------|--------|-------------------|--------------|
| off       | off    | `IDLE_READING`     | OCR → Merge Engine cleanup → memory/pointer map update → offline TTS plays |
| off       | on     | `MEANING_MODE`     | fingertip → pointed word/sentence → cleanup → AI Engine meaning → OLED |
| on        | off    | `UPDATE_POSITION`  | fingertip → pointed word → TTS pointer jumps there |
| on        | on     | `SCROLL_MODE`      | OLED scrolls the displayed meaning text (firmware handles the scroll itself on this transition) |

TTS is paused in every mode except `IDLE_READING`, and resumes from the exact
word index it left off on (or was just moved to), per spec.

## On the "Groq generates Python code" requirement
The spec describes Groq handing back Python code to keep the TTS reading
pointer correct. This implementation has Groq return a **JSON word-offset
map** instead (`merge_engine.py` → `update_memory_and_pointer_map`), which
`tts_engine.py` — ordinary, reviewed code — uses to track/resume position.
Executing model-generated code directly (`exec()`) would be a real
arbitrary-code-execution risk in a pipeline whose input is OCR text from a
camera; the JSON map achieves the same "pointer stays correct across pauses"
result without that risk.

## Setup
```bash
cd taletrace
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in your OWN rotated keys
python main.py
```
Flash `firmware/buttons_and_oled.ino` to the ESP32 (fill in your WiFi
creds there too) via Arduino IDE — needs the `U8g2` library and the ESP32
board package.

## Known gaps / things you'll need to tune on real hardware
- **`WEBCAM_DEVICE_INDEX`**: 0 is a guess. If another camera grabs index 0,
  bump it — `cv2.VideoCapture(1)`, etc. There's no cross-platform reliable
  way to auto-detect "the C270 specifically" from this script.
- **Fingertip → word mapping** (`gesture_engine.py`) assumes the OCR
  bounding boxes and the webcam frame are the same resolution/orientation.
  If your camera framing doesn't match what OCR.space measured, you'll need
  to add a coordinate transform (crop/scale factors) once you see it live.
- **`pyttsx3`'s per-word granularity** varies by OS voice engine; on some
  platforms it can only reliably pause between `say()` calls, not
  mid-word — which is why `tts_engine.py` speaks one word at a time rather
  than one sentence at a time (slightly less natural prosody, but pause
  points are always exact).
- **`BOOK_CONTEXT_HINT`** in `merge_engine.py` is a generic placeholder —
  put in the actual book/chapter title so the grammar-correction prompt has
  real context to correct against.
- Nothing here has been run against your actual camera/ESP32/book, since I
  don't have access to that hardware — the logic is complete and each
  module is independently testable, but expect to tune thresholds
  (`OCR_INTERVAL`, `DEBOUNCE_DELAY`, MediaPipe confidence) once it's on your
  desk.
