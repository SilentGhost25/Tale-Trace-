"""
Gesture Engine facade — connects the TaleTrace main loop to the
dual-tier gesture detector and adaptive cone word selector in gesture/.
"""
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from gesture.models import (
    OCRWord,
    SelectionResult,
    SelectionStatus,
    SelectionConfig,
)
from gesture import select_word
from gesture.finger_detector import detect_finger

log = logging.getLogger("taletrace.gesture")


@dataclass
class PointedWord:
    word: str
    context: str
    line: str = ""
    paragraph: str = ""
    confidence: float = 0.0
    selection_result: Optional[SelectionResult] = None


def quick_finger_check(frame: np.ndarray, config: Optional[SelectionConfig] = None) -> bool:
    """Fast local-only check: is any finger visible in the frame? (~50 ms, no network)."""
    if frame is None or frame.size == 0:
        return False
    cfg = config or SelectionConfig()
    finger = detect_finger(frame, cfg)
    return finger is not None


def find_pointed_word(
    frame: np.ndarray,
    ocr_result,
    config: Optional[SelectionConfig] = None,
) -> Optional[PointedWord]:
    """
    Detects any pointing finger (MediaPipe Tasks + HSV skin contour fallback),
    runs line-first direct touch and adaptive cone word selection,
    and returns PointedWord with target word and sentence/paragraph context.
    """
    if frame is None or ocr_result is None:
        return None

    ocr_words: List[OCRWord] = getattr(ocr_result, "words", [])
    if not ocr_words:
        return None

    result: SelectionResult = select_word(frame, ocr_words, config=config)

    if result.status == SelectionStatus.SUCCESS:
        log.info(
            "Target word identified: '%s' (confidence: %.2f, reason: %s)",
            result.selected_word,
            result.confidence,
            result.selection_reason,
        )
        return PointedWord(
            word=result.selected_word,
            context=result.context if result.context else result.selected_line,
            line=result.selected_line,
            paragraph=result.selected_paragraph,
            confidence=result.confidence,
            selection_result=result,
        )
    elif result.status == SelectionStatus.LOW_CONFIDENCE and result.selected_word:
        log.debug(
            "Low confidence word selected: '%s' (confidence: %.2f)",
            result.selected_word,
            result.confidence,
        )
        return PointedWord(
            word=result.selected_word,
            context=result.context if result.context else result.selected_line,
            line=result.selected_line,
            paragraph=result.selected_paragraph,
            confidence=result.confidence,
            selection_result=result,
        )

    log.debug("No pointed word detected: %s", result.status.value)
    return None