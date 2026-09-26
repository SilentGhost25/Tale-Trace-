"""
Standardized OCR Subsystem for TaleTrace.

Features:
- Primary Engine: Local Tesseract 5.5 (~240ms, 100% offline, zero network cost).
- Cloud Fallback: OCR.space Engine 2 (free tier: 500 requests/day).
- Excludes Google Cloud Vision and PaddleOCR per specifications.
- Produces standardized OCRWord objects compatible with the Gesture Engine.
"""
import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import requests

from config import settings
from gesture.models import OCRWord
from webcam_capture import preprocess_image

log = logging.getLogger("taletrace.ocr")

OCR_SPACE_URL = "https://api.ocr.space/parse/image"

import shutil
import time

# Configure Tesseract path
_tesseract_cmd = settings.tesseract_cmd
_tesseract_found = False

_candidate_paths = [
    _tesseract_cmd,
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]

for p in _candidate_paths:
    if p and os.path.exists(p):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = p
            _tesseract_cmd = p
            _tesseract_found = True
            log.info("Tesseract binary configured at: %s", p)
            break
        except ImportError:
            pass

if not _tesseract_found:
    which_path = shutil.which("tesseract")
    if which_path:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = which_path
            _tesseract_cmd = which_path
            _tesseract_found = True
            log.info("Tesseract binary found in PATH: %s", which_path)
        except ImportError:
            pass

if not _tesseract_found:
    log.warning(
        "Tesseract OCR executable not detected locally. TaleTrace will rely on OCR_SPACE_API_KEY from .env "
        "(or install Tesseract via 'winget install UB-Mannheim.TesseractOCR')."
    )

_last_tesseract_warn_time = 0.0


@dataclass
class OcrResult:
    text: str
    words: List[OCRWord]


def _encode_jpeg(frame: Union[np.ndarray, bytes]) -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for OCR")
    return buf.tobytes()


def ocr_image(frame: np.ndarray, use_cloud: bool = True) -> OcrResult:
    """
    Primary OCR entry point.
    Prioritizes OCR.space Cloud Engine 2 when API key is configured,
    with automatic offline fallback to local Tesseract OCR.
    """
    if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
        return OcrResult(text="", words=[])

    # 1. First Choice: OCR.space Cloud Engine (Engine 2)
    if settings.ocr_space_api_key and use_cloud:
        try:
            cloud_res = _ocr_space(frame)
            if cloud_res.text.strip():
                log.info("OCR.space cloud recognition successful: %d words extracted.", len(cloud_res.words))
                return cloud_res
            log.warning("OCR.space cloud produced no text; falling back to local Tesseract OCR...")
        except Exception as e:
            log.warning("OCR.space request failed (%s); falling back to local Tesseract OCR...", e)

    # 2. Fallback / Offline Choice: Local Tesseract OCR
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        proc_frame = preprocess_image(frame)
    else:
        proc_frame = frame

    return _ocr_tesseract(proc_frame)


def _ocr_tesseract(frame: np.ndarray) -> OcrResult:
    """Executes high-performance local Tesseract OCR with word bounding boxes."""
    global _last_tesseract_warn_time

    if not _tesseract_found:
        now = time.time()
        if now - _last_tesseract_warn_time > 30.0:
            log.warning(
                "Local Tesseract OCR executable not found on system. "
                "To fix: Add OCR_SPACE_API_KEY to .env or run 'winget install UB-Mannheim.TesseractOCR'."
            )
            _last_tesseract_warn_time = now
        return OcrResult(text="", words=[])

    try:
        import pytesseract
    except ImportError:
        log.error("pytesseract package not installed.")
        return OcrResult(text="", words=[])

    try:
        if isinstance(frame, (bytes, bytearray)):
            img = Image.open(io.BytesIO(frame))
        elif len(frame.shape) == 2:
            img = Image.fromarray(frame)
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Page segmentation mode 3: Fully automatic page segmentation without OSD; English only
        config = "--oem 3 --psm 3 -l eng"
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

        words: List[OCRWord] = []
        raw_words = []

        total_entries = len(data.get("text", []))
        for i in range(total_entries):
            w_text = data["text"][i].strip()
            if not w_text:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            conf_raw = data.get("conf", [100])[i]
            try:
                conf = float(conf_raw) / 100.0 if float(conf_raw) > 0 else 0.5
            except (ValueError, TypeError):
                conf = 0.5

            raw_words.append(w_text)
            words.append(
                OCRWord(
                    text=w_text,
                    bbox=(x, y, x + w, y + h),
                    center_x=(x + x + w) / 2.0,
                    center_y=(y + y + h) / 2.0,
                    confidence=conf,
                )
            )

        full_text = " ".join(raw_words).strip()
        return OcrResult(text=full_text, words=words)

    except Exception as e:
        log.error("Tesseract local OCR error: %s", e)
        return OcrResult(text="", words=[])


def _ocr_space(frame: Union[np.ndarray, bytes]) -> OcrResult:
    """Executes OCR.space Engine 2 cloud OCR (free tier) with timeout safeguards."""
    jpeg_bytes = _encode_jpeg(frame)
    payload = None

    # Retry up to 2 times on transient network error (503/timeout)
    for attempt in range(2):
        try:
            resp = requests.post(
                OCR_SPACE_URL,
                files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
                data={
                    "apikey": settings.ocr_space_api_key,
                    "OCREngine": 2,
                    "isOverlayRequired": True,
                    "scale": True,
                },
                timeout=8,
            )
            if resp.status_code == 503 and attempt == 0:
                time.sleep(1.0)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as e:
            if attempt == 1:
                log.warning("OCR.space request attempt failed: %s", e)
                return OcrResult(text="", words=[])
            time.sleep(0.5)

    if not payload:
        return OcrResult(text="", words=[])

    if payload.get("IsErroredOnProcessing"):
        log.error("OCR.space API error: %s", payload.get("ErrorMessage"))
        return OcrResult(text="", words=[])

    parsed = payload.get("ParsedResults", [])
    if not parsed:
        return OcrResult(text="", words=[])

    result = parsed[0]
    text = result.get("ParsedText", "").strip()

    words: List[OCRWord] = []
    overlay = result.get("TextOverlay", {})
    for line in overlay.get("Lines", []):
        for w in line.get("Words", []):
            w_text = w.get("WordText", "").strip()
            if not w_text:
                continue
            left = int(w.get("Left", 0))
            top = int(w.get("Top", 0))
            width = int(w.get("Width", 0))
            height = int(w.get("Height", 0))

            words.append(
                OCRWord(
                    text=w_text,
                    bbox=(left, top, left + width, top + height),
                    center_x=(left + left + width) / 2.0,
                    center_y=(top + top + height) / 2.0,
                    confidence=0.9,
                )
            )

    return OcrResult(text=text, words=words)
