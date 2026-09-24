import time
import numpy as np
from typing import Optional, List
from dataclasses import replace
from .models import SelectionConfig, SelectionResult, OCRWord, SelectionStatus, FingerPoint
from .finger_detector import detect_finger
from .word_selector import select_intended_word

API_VERSION = "1.0"


def select_word(
    image: np.ndarray,
    ocr_words: List[OCRWord],
    config: Optional[SelectionConfig] = None,
    **kwargs,
) -> SelectionResult:
    """
    Unified entry point for Gesture Engine Word Selection.
    Always returns an immutable SelectionResult with diagnostic status.
    """
    start_time = time.perf_counter()
    cfg = config or SelectionConfig()

    if image is None or image.size == 0:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return SelectionResult(
            status=SelectionStatus.NO_FINGER,
            selection_time_ms=duration_ms,
        )

    img_h, img_w = image.shape[:2]

    if not ocr_words:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return SelectionResult(
            status=SelectionStatus.OCR_EMPTY,
            image_size=(img_w, img_h),
            selection_time_ms=duration_ms,
        )

    finger = detect_finger(image, cfg)
    if not finger:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return SelectionResult(
            status=SelectionStatus.NO_FINGER,
            image_size=(img_w, img_h),
            selection_time_ms=duration_ms,
        )

    result = select_intended_word(finger, ocr_words, cfg)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return replace(result, image_size=(img_w, img_h), selection_time_ms=duration_ms)
