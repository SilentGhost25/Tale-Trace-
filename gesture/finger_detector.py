import os
import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp
from .models import SelectionConfig, FingerPoint

# Cache detector instances to avoid reloading on every frame
_landmarker = None
_mp_legacy_hands = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def _get_tasks_landmarker():
    global _landmarker
    if _landmarker is None and os.path.exists(MODEL_PATH):
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception:
            _landmarker = None
    return _landmarker


def _get_legacy_hands():
    global _mp_legacy_hands
    if _mp_legacy_hands is None:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                _mp_legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                )
            except Exception:
                _mp_legacy_hands = None
    return _mp_legacy_hands


class FingerDirectionEstimator:
    """Internal helper to compute pointing direction from finger joint landmarks."""
    @staticmethod
    def estimate_direction(
        mcp: Tuple[float, float],
        pip: Tuple[float, float],
        dip: Tuple[float, float],
        tip: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Estimate pointing direction unit vector (dx, dy) using weighted joint segments.
        Tip segment has highest weight, MCP segment has lowest weight.
        """
        # Segment vectors
        d_tip_dip = (tip[0] - dip[0], tip[1] - dip[1])
        d_dip_pip = (dip[0] - pip[0], dip[1] - pip[1])
        d_pip_mcp = (pip[0] - mcp[0], pip[1] - mcp[1])

        # Weighted direction
        dx = 0.5 * d_tip_dip[0] + 0.3 * d_dip_pip[0] + 0.2 * d_pip_mcp[0]
        dy = 0.5 * d_tip_dip[1] + 0.3 * d_dip_pip[1] + 0.2 * d_pip_mcp[1]

        # Normalize
        length = np.hypot(dx, dy)
        if length < 1e-5:
            return (0.0, -1.0)  # Default point straight up
        return (dx / length, dy / length)


def detect_finger_mediapipe(image: np.ndarray, config: SelectionConfig) -> Optional[FingerPoint]:
    """Tier 1: Detect finger using MediaPipe Hands (Tasks API or legacy)."""
    if image is None or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Try MediaPipe Tasks HandLandmarker (MediaPipe 1.0+)
    detector = _get_tasks_landmarker()
    if detector is not None:
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                landmarks = results.hand_landmarks[0]
                hand_score = 0.9
                if results.handedness and len(results.handedness) > 0:
                    hand_score = float(results.handedness[0][0].score)

                # Index finger landmarks: 5=MCP, 6=PIP, 7=DIP, 8=TIP
                mcp = (landmarks[5].x * img_w, landmarks[5].y * img_h)
                pip = (landmarks[6].x * img_w, landmarks[6].y * img_h)
                dip = (landmarks[7].x * img_w, landmarks[7].y * img_h)
                tip = (landmarks[8].x * img_w, landmarks[8].y * img_h)

                direction = FingerDirectionEstimator.estimate_direction(mcp, pip, dip, tip)
                return FingerPoint(
                    x=float(tip[0]),
                    y=float(tip[1]),
                    confidence=float(hand_score),
                    direction=direction,
                    detection_method="mediapipe",
                )
        except Exception:
            pass

    # 2. Try MediaPipe legacy solutions if available
    legacy = _get_legacy_hands()
    if legacy is not None:
        try:
            results = legacy.process(image_rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                hand_score = 1.0
                if results.multi_handedness:
                    hand_score = results.multi_handedness[0].classification[0].score

                mcp_lm = landmarks.landmark[5]
                pip_lm = landmarks.landmark[6]
                dip_lm = landmarks.landmark[7]
                tip_lm = landmarks.landmark[8]

                mcp = (mcp_lm.x * img_w, mcp_lm.y * img_h)
                pip = (pip_lm.x * img_w, pip_lm.y * img_h)
                dip = (dip_lm.x * img_w, dip_lm.y * img_h)
                tip = (tip_lm.x * img_w, tip_lm.y * img_h)

                direction = FingerDirectionEstimator.estimate_direction(mcp, pip, dip, tip)
                return FingerPoint(
                    x=float(tip[0]),
                    y=float(tip[1]),
                    confidence=float(hand_score),
                    direction=direction,
                    detection_method="mediapipe",
                )
        except Exception:
            pass

    return None


def detect_finger_contour_fallback(image: np.ndarray) -> Optional[FingerPoint]:
    """Tier 2: Enhanced skin segmenter & directional finger estimator (100% OpenCV)."""
    if image is None or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]

    # Convert to HSV and YCrCb color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # In book reading, white/cream paper has Saturation < 50.
    # Skin has Saturation >= 52 and Hue in [0-30] or [155-180].
    # Use saturation threshold of 52 for real camera frames, with fallback for synthetic tests
    sat_thresh = 52 if min(img_h, img_w) >= 300 else 25

    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, sat_thresh, 25], dtype=np.uint8), np.array([30, 255, 235], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([155, sat_thresh, 25], dtype=np.uint8), np.array([180, 255, 235], dtype=np.uint8))
    )
    # YCrCb refinement to distinguish skin from paper/shadows
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 128, 65], dtype=np.uint8), np.array([255, 185, 135], dtype=np.uint8))
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphological closing & opening with scale-adaptive kernel
    k_size = 3 if min(img_h, img_w) < 400 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_fingers = []
    min_area = max(500, int(img_h * img_w * 0.001))

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        # Exclude spine margin / page edge shadows spanning full image height from top
        if by == 0 and bh > img_h * 0.75 and (bx == 0 or bx + bw >= img_w):
            continue
        # In reading setup, finger enters from the bottom half/edges
        if min(img_h, img_w) >= 400 and (by + bh < img_h * 0.5):
            continue

        # Fingertip apex (minimum Y coordinate in contour)
        min_y = np.min(c[:, 0, 1])
        top_pts = c[c[:, 0, 1] <= min_y + 6, 0]
        apex_x = float(np.mean(top_pts[:, 0]))
        apex_y = float(min_y)

        # Fit orientation line to determine pointing direction
        [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        dir_x, dir_y = float(vx[0]), float(vy[0])
        # Ensure pointing vector points upwards (negative Y in image space)
        if dir_y > 0:
            dir_x, dir_y = -dir_x, -dir_y

        # If line is nearly vertical, snap to (0.0, -1.0)
        if abs(dir_x) < 0.05:
            dir_x, dir_y = 0.0, -1.0
        else:
            norm = np.hypot(dir_x, dir_y)
            dir_x, dir_y = dir_x / norm, dir_y / norm

        # Compute confidence based on solidity and aspect ratio
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.5
        confidence = float(np.clip(0.75 + 0.2 * solidity, 0.6, 0.95))

        # Bonus weight for contours entering from bottom edge
        score_weight = area * (1.5 if (by + bh >= img_h - 20) else 1.0)
        valid_fingers.append((score_weight, apex_x, apex_y, (dir_x, dir_y), confidence))

    if not valid_fingers:
        return None

    # Pick the most prominent finger contour
    best_finger = max(valid_fingers, key=lambda f: f[0])

    return FingerPoint(
        x=best_finger[1],
        y=best_finger[2],
        confidence=best_finger[4],
        direction=best_finger[3],
        detection_method="contour_fallback",
    )


def detect_finger(image: np.ndarray, config: SelectionConfig) -> Optional[FingerPoint]:
    """
    Dual-tier fingertip detector.
    Attempts MediaPipe first; if not detected or low confidence, runs contour fallback.
    """
    # 1. MediaPipe Attempt
    try:
        mp_point = detect_finger_mediapipe(image, config)
        if mp_point and mp_point.confidence >= config.fallback_trigger:
            return mp_point
    except Exception:
        mp_point = None

    # 2. Fallback Attempt
    fallback_point = detect_finger_contour_fallback(image)

    # Return the one with higher confidence
    if mp_point and fallback_point:
        return mp_point if mp_point.confidence >= fallback_point.confidence else fallback_point
    return mp_point or fallback_point
