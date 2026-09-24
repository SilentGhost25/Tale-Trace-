import cv2
import os
from gesture import select_word
from gesture.visualizer import draw_overlay
from gesture.models import SelectionConfig
from ocr_engine import ocr_image

img_path = r"C:\Users\dhrut\Desktop\WhatsApp Image 2026-09-24 at 11.51.01 AM.jpeg"
image = cv2.imread(img_path)

if image is None:
    print(f"Error: Could not load image from {img_path}")
    exit(1)

ocr_res = ocr_image(image)
cfg = SelectionConfig()
result = select_word(image, ocr_res.words, cfg)

print("=== GESTURE ENGINE OUTPUT ===")
print(f"Status: {result.status.value}")
print(f"Selected Word: '{result.selected_word}'")
print(f"Selected Word BBox: {result.selected_word_bbox}")
print(f"Selected Line: '{result.selected_line}'")
print(f"Sentence Context: '{result.context}'")
print(f"Confidence Score: {result.confidence:.4f}")
if result.finger_point:
    print(f"Fingertip Coordinate: ({result.finger_point.x:.1f}, {result.finger_point.y:.1f})")
    print(f"Pointing Direction: {result.finger_point.direction}")
    print(f"Detection Method: {result.finger_point.detection_method}")
print(f"Selection Reason: {result.selection_reason}")
print(f"Processing Time: {result.selection_time_ms:.1f} ms")

out_dir = r"C:\Users\dhrut\.gemini\antigravity-ide\brain\5a5b8dc4-28df-44ac-b7bd-d897ba1b023b"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "gesture_output.jpg")
annotated = draw_overlay(image, result, mode="developer")
cv2.imwrite(out_path, annotated)
print(f"Annotated visual overlay saved to: {out_path}")
