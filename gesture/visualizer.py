import cv2
import numpy as np
from typing import Optional
from .models import SelectionResult, SelectionStatus

def draw_overlay(image: np.ndarray, result: SelectionResult, mode: str = "basic") -> np.ndarray:
    """
    Draw debugging and visualization overlay on a copy of the input image.
    Supports 'basic', 'developer', and 'debug' modes.
    """
    viz_img = image.copy()
    img_h, img_w = viz_img.shape[:2]
    
    # Check if we have finger point coordinates
    if not result.finger_point:
        # Draw status notice at the top
        text = f"Status: {result.status.value.upper()}"
        cv2.putText(viz_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return viz_img
        
    finger = result.finger_point
    fx, fy = int(finger.x), int(finger.y)
    
    # ----------------------------------------------------
    # DEBUG Mode: Draw candidates, line clusters, and vectors
    # ----------------------------------------------------
    if mode == "debug":
        # Draw all candidate scores
        for i, candidate in enumerate(result.candidate_scores):
            w = candidate.word
            # Draw box around candidate (light gray/yellow for runners-up)
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.rectangle(viz_img, (w.bbox[0], w.bbox[1]), (w.bbox[2], w.bbox[3]), color, thickness)
            
            # Print score next to word (offset to avoid overlapping)
            score_text = f"{candidate.total_score:.2f}"
            cv2.putText(viz_img, score_text, (w.bbox[0], w.bbox[1] - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
        # Draw pointing direction vector if available
        if finger.direction:
            dx, dy = finger.direction
            # Draw pointing vector arrow of 100px length starting from fingertip
            v_len = 100
            end_x = int(fx + dx * v_len)
            end_y = int(fy + dy * v_len)
            # Arrow in magenta
            cv2.arrowedLine(viz_img, (fx, fy), (end_x, end_y), (255, 0, 255), 2, tipLength=0.2)
            cv2.putText(viz_img, "Pointing Vector", (end_x + 5, end_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # ----------------------------------------------------
    # DEVELOPER & DEBUG Modes: Draw Search Region & Target Line
    # ----------------------------------------------------
    if mode in ("developer", "debug"):
        # We need to reconstruct the search region polygon coordinates
        # Since we don't store the exact polygon in SelectionResult, we can reconstruct it from the code:
        from .word_selector import get_search_polygon, group_ocr_words, SelectionConfig
        
        # Calculate search polygon outline
        # Since we don't have the original OCRWords here, we can estimate polygon based on selected word box if available
        # or average word height. If we have candidates, we use their coordinates.
        if result.candidate_scores:
            words = [c.word for c in result.candidate_scores]
            avg_word_width = np.mean([w.bbox[2] - w.bbox[0] for w in words])
            avg_line_height = np.mean([w.bbox[3] - w.bbox[1] for w in words])
            # Construct a dummy config for visualization sizing
            cfg = SelectionConfig()
            search_poly = get_search_polygon(finger, avg_line_height, avg_word_width, cfg)
            
            # Draw search polygon outline in Blue
            pts = np.array(search_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(viz_img, "Search Zone", (pts[2][0][0], pts[2][0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Highlight target line bounding box in cyan
            # Find the best line bounding box by merging the word bboxes in candidates
            best_line_words = [w for w in words if w.line_index == result.line_index]
            if best_line_words:
                lx_min = min(w.bbox[0] for w in best_line_words)
                ly_min = min(w.bbox[1] for w in best_line_words)
                lx_max = max(w.bbox[2] for w in best_line_words)
                ly_max = max(w.bbox[3] for w in best_line_words)
                cv2.rectangle(viz_img, (lx_min, ly_min), (lx_max, ly_max), (255, 255, 0), 1)
                cv2.putText(viz_img, f"Line {result.line_index}", (lx_min, ly_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ----------------------------------------------------
    # BASIC Mode (Present in all modes): Draw fingertip & selected word
    # ----------------------------------------------------
    # Draw fingertip (red filled circle)
    cv2.circle(viz_img, (fx, fy), 8, (0, 0, 255), -1)
    cv2.circle(viz_img, (fx, fy), 10, (255, 255, 255), 1)  # White border
    
    if result.status == SelectionStatus.SUCCESS or result.status == SelectionStatus.LOW_CONFIDENCE:
        # Highlight selected word box in green
        if result.selected_word_bbox:
            bx_min, by_min, bx_max, by_max = result.selected_word_bbox
            cv2.rectangle(viz_img, (bx_min, by_min), (bx_max, by_max), (0, 255, 0), 3)
            
            # Word label directly above selected word box
            label = f"'{result.selected_word}'"
            if result.status == SelectionStatus.LOW_CONFIDENCE:
                label += " (Low Conf)"
            cv2.putText(viz_img, label, (bx_min, by_min - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
    # Draw a status board at the top left of the image
    cv2.rectangle(viz_img, (10, 10), (350, 100), (0, 0, 0), -1)  # Black backdrop
    
    status_color = (0, 255, 0) if result.status == SelectionStatus.SUCCESS else (
        (0, 255, 255) if result.status == SelectionStatus.LOW_CONFIDENCE else (0, 0, 255)
    )
    cv2.putText(viz_img, f"Status: {result.status.value.upper()}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(viz_img, f"Confidence: {result.confidence:.2f}", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(viz_img, f"Time: {result.selection_time_ms:.1f} ms", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if result.status in (SelectionStatus.SUCCESS, SelectionStatus.LOW_CONFIDENCE):
        cv2.putText(viz_img, f"Word: {result.selected_word}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
    return viz_img
