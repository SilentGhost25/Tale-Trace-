from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


class SelectionStatus(Enum):
    SUCCESS = "success"
    NO_FINGER = "no_finger_detected"
    LOW_CONFIDENCE = "low_selection_confidence"
    NO_WORD_FOUND = "no_word_in_search_region"
    OCR_EMPTY = "ocr_data_empty"


class SelectionStrategy(Enum):
    STATIC_BOX = "static_box"
    DIRECTION_CONE = "direction_cone"
    AUTO = "auto"
    TOUCH = "touch"
    POINT = "point"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class FingerPoint:
    x: float
    y: float
    confidence: float
    direction: Optional[Tuple[float, float]] = None  # pointing unit vector (dx, dy)
    detection_method: str = "mediapipe"              # "mediapipe" | "contour_fallback"


@dataclass(frozen=True)
class OCRWord:
    text: str
    bbox: Tuple[int, int, int, int]                  # (x_min, y_min, x_max, y_max)
    center_x: float
    center_y: float
    confidence: float = 1.0
    word_index: int = -1
    line_index: int = -1
    paragraph_index: int = -1


@dataclass(frozen=True)
class TextLine:
    words: List[OCRWord]
    y_center: float
    bbox: Tuple[int, int, int, int]
    line_index: int
    paragraph_index: int = -1


@dataclass(frozen=True)
class SelectionConfig:
    selection_strategy: SelectionStrategy = SelectionStrategy.AUTO

    # Search region (relative ratios to average line height)
    search_height_above: float = 1.5
    search_height_below: float = 0.3
    search_width_ratio: float = 3.0

    # Scoring weights
    vertical_bias: float = 0.30
    horizontal_weight: float = 0.30
    direction_weight: float = 0.25
    overlap_weight: float = 0.15

    # Thresholds
    line_cluster_tolerance: float = 0.5
    confidence_threshold: float = 0.4
    mediapipe_confidence: float = 0.5
    fallback_trigger: float = 0.75

    use_direction: bool = True


@dataclass(frozen=True)
class CoordinateTransformer:
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def transform(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x) + self.offset_x, (y * self.scale_y) + self.offset_y

    def inverse_transform(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.offset_x) / self.scale_x, (y - self.offset_y) / self.scale_y


@dataclass(frozen=True)
class ScoredCandidate:
    word: OCRWord
    vertical_score: float
    horizontal_score: float
    direction_score: float
    overlap_score: float
    total_score: float
    scoring_reason: str


@dataclass(frozen=True)
class SelectionResult:
    status: SelectionStatus
    selected_word: str = ""
    selected_line: str = ""
    selected_line_words: List[str] = field(default_factory=list)
    selected_paragraph: str = ""
    context: str = ""
    confidence: float = 0.0
    finger_point: Optional[FingerPoint] = None
    selected_word_bbox: Optional[Tuple[int, int, int, int]] = None
    word_index: int = -1
    line_index: int = -1
    paragraph_index: int = -1
    candidate_scores: List[ScoredCandidate] = field(default_factory=list)
    selection_reason: str = ""

    # Debugging & Performance Metadata
    image_size: Tuple[int, int] = (0, 0)
    transform_applied: str = "identity"
    selection_time_ms: float = 0.0
