import pytest
import numpy as np
from gesture_engine.models import (
    FingerPoint,
    OCRWord,
    SelectionConfig,
    SelectionStatus,
    SelectionStrategy
)
from gesture_engine.word_selector import (
    group_ocr_words,
    get_search_polygon,
    select_intended_word
)

def test_dynamic_line_clustering():
    # 4 words in 2 different lines
    ocr_words = [
        # Line 0
        OCRWord(text="Hello", bbox=(10, 20, 60, 40), center_x=35.0, center_y=30.0),
        OCRWord(text="World", bbox=(80, 20, 130, 40), center_x=105.0, center_y=30.0),
        # Line 1 (vertically spaced by 50px)
        OCRWord(text="Python", bbox=(10, 70, 70, 90), center_x=40.0, center_y=80.0),
        OCRWord(text="Rules", bbox=(90, 70, 140, 90), center_x=115.0, center_y=80.0),
    ]
    
    cfg = SelectionConfig(line_cluster_tolerance=0.5)
    lines = group_ocr_words(ocr_words, cfg)
    
    assert len(lines) == 2
    
    # First cluster (Line 0)
    assert lines[0].line_index == 0
    assert len(lines[0].words) == 2
    assert lines[0].words[0].text == "Hello"
    assert lines[0].words[1].text == "World"
    assert lines[0].words[0].line_index == 0
    assert lines[0].words[1].line_index == 0
    
    # Second cluster (Line 1)
    assert lines[1].line_index == 1
    assert len(lines[1].words) == 2
    assert lines[1].words[0].text == "Python"
    assert lines[1].words[1].text == "Rules"

def test_paragraph_grouping():
    # Test gap-based paragraph grouping. We will create 3 lines,
    # Line 0 and Line 1 are close, Line 2 is far away.
    ocr_words = [
        OCRWord(text="LineOne", bbox=(10, 10, 80, 30), center_x=45.0, center_y=20.0),
        OCRWord(text="LineTwo", bbox=(10, 40, 80, 60), center_x=45.0, center_y=50.0),
        OCRWord(text="LineThree", bbox=(10, 120, 80, 140), center_x=45.0, center_y=130.0),
    ]
    
    cfg = SelectionConfig()
    lines = group_ocr_words(ocr_words, cfg)
    
    assert len(lines) == 3
    # Check paragraph indices
    # LineOne and LineTwo are spaced by 30px
    # LineTwo and LineThree are spaced by 80px. Since 80px > 1.8 * 30px (which is 54px),
    # there should be a paragraph break between Line 1 and Line 2!
    assert lines[0].paragraph_index == 0
    assert lines[1].paragraph_index == 0
    assert lines[2].paragraph_index == 1

def test_search_polygon_projection():
    # Finger point pointing right
    finger = FingerPoint(x=100.0, y=100.0, confidence=1.0, direction=(1.0, 0.0), detection_method="mediapipe")
    
    # average word size: height=20, width=50
    poly = get_search_polygon(finger, avg_line_height=20.0, avg_word_width=50.0, config=SelectionConfig(selection_strategy=SelectionStrategy.DIRECTION_CONE))
    
    # check that points are projected along x axis (dx=1, dy=0)
    # orth vector is nx=-dy=0, ny=dx=1
    # end point center at x0 + L*dx = 100 + 1.5*20 * 1 = 130
    # base center at x0 - buffer*dx = 100 - 0.3*20 * 1 = 94
    # Check that Y values are symmetric around 100.0
    assert np.mean([pt[1] for pt in poly]) == pytest.approx(100.0, abs=1e-5)
    # Check that X extends from 94 to 130
    x_coords = [pt[0] for pt in poly]
    assert min(x_coords) == pytest.approx(94.0, abs=1e-5)
    assert max(x_coords) == pytest.approx(130.0, abs=1e-5)

def test_word_selector_selection():
    # Set up OCR words representing a paragraph
    ocr_words = [
        # Line 0: "Two topics impact everyone."
        OCRWord(text="Two", bbox=(100, 85, 150, 115), center_x=125.0, center_y=100.0),
        OCRWord(text="topics", bbox=(170, 85, 260, 115), center_x=215.0, center_y=100.0),
        OCRWord(text="impact", bbox=(280, 85, 370, 115), center_x=325.0, center_y=100.0),
        OCRWord(text="everyone.",  bbox=(390, 85, 520, 115), center_x=455.0, center_y=100.0),
        
        # Line 1: "Whether you are interested"
        OCRWord(text="Whether", bbox=(100, 135, 220, 165), center_x=160.0, center_y=150.0),
        OCRWord(text="you", bbox=(240, 135, 290, 165), center_x=265.0, center_y=150.0),
        OCRWord(text="are", bbox=(310, 135, 350, 165), center_x=330.0, center_y=150.0),
        OCRWord(text="interested", bbox=(370, 135, 500, 165), center_x=435.0, center_y=150.0),
    ]
    
    # Finger pointing at "you" (Line 1, center_x=265, center_y=150)
    # We place the finger slightly below "you" at (265, 175) pointing straight up
    finger = FingerPoint(x=265.0, y=175.0, confidence=0.9, direction=(0.0, -1.0))
    
    cfg = SelectionConfig(confidence_threshold=0.3)
    res = select_intended_word(finger, ocr_words, cfg)
    
    assert res.status == SelectionStatus.SUCCESS
    assert res.selected_word == "you"
    assert res.selected_line == "Whether you are interested"
    assert res.selected_line_words == ["Whether", "you", "are", "interested"]
    assert res.line_index == 1
    
    # Context should be the sentence containing the word.
    # In this case, since there is no sentence-ending punctuation on Line 1,
    # the sentence reconstruction might group it based on paragraph bounds.
    # Let's check that context is reconstructed
    assert "you" in res.context
    
    # Verify confidence is a float between 0 and 1
    assert 0.0 <= res.confidence <= 1.0

def test_word_selector_no_finger():
    # Verify edge case where selector behaves with far off point
    ocr_words = [
        OCRWord(text="word", bbox=(10, 10, 50, 30), center_x=30.0, center_y=20.0)
    ]
    # Finger is far away at (500, 500)
    finger = FingerPoint(x=500.0, y=500.0, confidence=0.9)
    res = select_intended_word(finger, ocr_words, SelectionConfig())
    assert res.status == SelectionStatus.NO_WORD_FOUND


def test_selection_strategy_touch():
    ocr_words = [
        OCRWord(text="first", bbox=(10, 10, 50, 30), center_x=30.0, center_y=20.0),
        OCRWord(text="second", bbox=(10, 40, 50, 60), center_x=30.0, center_y=50.0),
    ]
    # Finger touches exactly at the "second" word center
    finger = FingerPoint(x=30.0, y=50.0, confidence=0.9, direction=(0.0, -1.0))
    cfg = SelectionConfig(selection_strategy=SelectionStrategy.TOUCH)
    res = select_intended_word(finger, ocr_words, cfg)
    assert res.selected_word == "second"


def test_selection_strategy_point():
    ocr_words = [
        OCRWord(text="above", bbox=(10, 10, 50, 30), center_x=30.0, center_y=20.0),
        OCRWord(text="below", bbox=(10, 40, 50, 60), center_x=30.0, center_y=50.0),
    ]
    # Finger is near "below" but points straight up at "above"
    finger = FingerPoint(x=30.0, y=45.0, confidence=0.9, direction=(0.0, -1.0))
    cfg = SelectionConfig(selection_strategy=SelectionStrategy.POINT)
    res = select_intended_word(finger, ocr_words, cfg)
    assert res.selected_word == "above"


def test_selection_strategy_hybrid():
    ocr_words = [
        OCRWord(text="left", bbox=(10, 10, 50, 30), center_x=30.0, center_y=20.0),
        OCRWord(text="right", bbox=(60, 10, 100, 30), center_x=80.0, center_y=20.0),
    ]
    # Finger is at (35, 35) pointing slightly right-up
    finger = FingerPoint(x=35.0, y=35.0, confidence=0.8, direction=(0.8, -0.6))
    cfg = SelectionConfig(selection_strategy=SelectionStrategy.HYBRID)
    res = select_intended_word(finger, ocr_words, cfg)
    assert res.selected_word in ("left", "right")

