import cv2
import numpy as np
import pytest
from gesture_engine.models import SelectionConfig, FingerPoint
from gesture_engine.finger_detector import (
    FingerDirectionEstimator,
    detect_finger_contour_fallback,
    detect_finger
)

def test_direction_estimator():
    # MCP, PIP, DIP, TIP pointing straight up
    mcp = (100.0, 200.0)
    pip = (100.0, 150.0)
    dip = (100.0, 100.0)
    tip = (100.0, 50.0)
    
    direction = FingerDirectionEstimator.estimate_direction(mcp, pip, dip, tip)
    assert direction[0] == pytest.approx(0.0, abs=1e-5)
    assert direction[1] == pytest.approx(-1.0, abs=1e-5)
    
    # Pointing diagonally up-right
    # (x increases, y decreases)
    mcp = (0.0, 0.0)
    pip = (10.0, -10.0)
    dip = (20.0, -20.0)
    tip = (30.0, -30.0)
    direction = FingerDirectionEstimator.estimate_direction(mcp, pip, dip, tip)
    assert direction[0] == pytest.approx(0.707106, abs=1e-3)
    assert direction[1] == pytest.approx(-0.707106, abs=1e-3)

def test_contour_fallback_black_image():
    # An all-black image should return None (no skin detected)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detect_finger_contour_fallback(img)
    assert res is None

def test_contour_fallback_skin_blob():
    # Create an HSV image and draw a skin-colored triangle pointing up
    hsv_canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Skin HSV color (e.g. Hue=10, Sat=150, Val=200)
    skin_color = (10, 150, 200)
    
    # Vertices of triangle pointing up: Tip at (100, 50), Base at (80, 150) and (120, 150)
    pts = np.array([[100, 50], [80, 150], [120, 150]], dtype=np.int32)
    cv2.fillConvexPoly(hsv_canvas, pts, skin_color)
    
    # Convert to BGR
    bgr_img = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
    
    # Detect fingertip
    res = detect_finger_contour_fallback(bgr_img)
    assert res is not None
    assert res.detection_method == "contour_fallback"
    # Tip of the convex hull should be at or near (100, 50)
    assert res.x == pytest.approx(100.0, abs=5.0)
    assert res.y == pytest.approx(50.0, abs=10.0)
    assert res.direction == (0.0, -1.0)
