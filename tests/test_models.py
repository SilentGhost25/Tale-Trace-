from gesture_engine.models import (
    SelectionStatus,
    SelectionStrategy,
    FingerPoint,
    OCRWord,
    CoordinateTransformer,
    SelectionConfig,
    SelectionResult
)

def test_models_instantiation():
    # Test Enums
    assert SelectionStatus.SUCCESS.value == "success"
    assert SelectionStrategy.AUTO.value == "auto"

    # Test FingerPoint
    fp = FingerPoint(x=100.0, y=150.0, confidence=0.9, direction=(0.0, -1.0))
    assert fp.x == 100.0
    assert fp.y == 150.0
    assert fp.confidence == 0.9
    assert fp.direction == (0.0, -1.0)
    assert fp.detection_method == "mediapipe"

    # Test OCRWord
    word = OCRWord(text="test", bbox=(10, 20, 50, 40), center_x=30.0, center_y=30.0)
    assert word.text == "test"
    assert word.bbox == (10, 20, 50, 40)
    assert word.center_x == 30.0
    assert word.center_y == 30.0
    assert word.word_index == -1

    # Test CoordinateTransformer
    tx = CoordinateTransformer(scale_x=2.0, scale_y=0.5, offset_x=10.0, offset_y=-5.0)
    trans = tx.transform(10.0, 100.0)
    assert trans == (30.0, 45.0)
    orig = tx.inverse_transform(30.0, 45.0)
    assert orig == (10.0, 100.0)
