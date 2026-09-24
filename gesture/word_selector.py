import cv2
import numpy as np
from typing import List, Optional, Tuple
from .models import (
    FingerPoint,
    OCRWord,
    TextLine,
    SelectionConfig,
    SelectionResult,
    SelectionStatus,
    SelectionStrategy,
    ScoredCandidate
)

def group_ocr_words(ocr_words: List[OCRWord], config: SelectionConfig) -> List[TextLine]:
    """Group OCRWords into lines and paragraphs. Respects pre-assigned line_index if present."""
    # Check if lines are pre-grouped
    has_pregrouped = all(w.line_index != -1 for w in ocr_words)
    
    if has_pregrouped:
        # Group by pre-assigned line_index
        lines_map = {}
        for w in ocr_words:
            lines_map.setdefault(w.line_index, []).append(w)
        
        text_lines = []
        for line_idx, words in sorted(lines_map.items()):
            # Sort words horizontally
            words_sorted = sorted(words, key=lambda x: x.center_x)
            # Re-create to fill word_index if not set
            updated_words = []
            for i, w in enumerate(words_sorted):
                updated_words.append(OCRWord(
                    text=w.text,
                    bbox=w.bbox,
                    center_x=w.center_x,
                    center_y=w.center_y,
                    confidence=w.confidence,
                    word_index=i if w.word_index == -1 else w.word_index,
                    line_index=w.line_index,
                    paragraph_index=w.paragraph_index
                ))
            
            y_centers = [w.center_y for w in updated_words]
            y_center = np.mean(y_centers) if y_centers else 0.0
            
            x_min = min(w.bbox[0] for w in updated_words)
            y_min = min(w.bbox[1] for w in updated_words)
            x_max = max(w.bbox[2] for w in updated_words)
            y_max = max(w.bbox[3] for w in updated_words)
            
            text_lines.append(TextLine(
                words=updated_words,
                y_center=y_center,
                bbox=(x_min, y_min, x_max, y_max),
                line_index=line_idx,
                paragraph_index=updated_words[0].paragraph_index
            ))
        return text_lines

    else:
        # Dynamic clustering based on vertical spacing
        if not ocr_words:
            return []
        
        # Calculate avg word height
        avg_height = np.mean([w.bbox[3] - w.bbox[1] for w in ocr_words])
        tolerance = config.line_cluster_tolerance * avg_height
        
        # Sort words by vertical center
        sorted_by_y = sorted(ocr_words, key=lambda w: w.center_y)
        
        clusters = []
        for w in sorted_by_y:
            placed = False
            for cluster in clusters:
                avg_y = np.mean([item.center_y for item in cluster])
                if abs(w.center_y - avg_y) <= tolerance:
                    cluster.append(w)
                    placed = True
                    break
            if not placed:
                clusters.append([w])
                
        # Sort clusters vertically by their average Y center
        clusters = sorted(clusters, key=lambda c: np.mean([w.center_y for w in c]))
        
        text_lines = []
        for line_idx, cluster in enumerate(clusters):
            # Sort words in line horizontally
            words_sorted = sorted(cluster, key=lambda w: w.center_x)
            
            updated_words = []
            for i, w in enumerate(words_sorted):
                updated_words.append(OCRWord(
                    text=w.text,
                    bbox=w.bbox,
                    center_x=w.center_x,
                    center_y=w.center_y,
                    confidence=w.confidence,
                    word_index=i,
                    line_index=line_idx,
                    paragraph_index=-1
                ))
                
            y_centers = [w.center_y for w in updated_words]
            y_center = np.mean(y_centers) if y_centers else 0.0
            
            x_min = min(w.bbox[0] for w in updated_words)
            y_min = min(w.bbox[1] for w in updated_words)
            x_max = max(w.bbox[2] for w in updated_words)
            y_max = max(w.bbox[3] for w in updated_words)
            
            text_lines.append(TextLine(
                words=updated_words,
                y_center=y_center,
                bbox=(x_min, y_min, x_max, y_max),
                line_index=line_idx,
                paragraph_index=-1
            ))
            
        # Perform gap-based paragraph grouping
        if len(text_lines) > 1:
            p_idx = 0
            updated_lines = []
            
            first_line = text_lines[0]
            updated_words = [
                OCRWord(
                    text=w.text,
                    bbox=w.bbox,
                    center_x=w.center_x,
                    center_y=w.center_y,
                    confidence=w.confidence,
                    word_index=w.word_index,
                    line_index=w.line_index,
                    paragraph_index=p_idx
                ) for w in first_line.words
            ]
            updated_lines.append(TextLine(
                words=updated_words,
                y_center=first_line.y_center,
                bbox=first_line.bbox,
                line_index=first_line.line_index,
                paragraph_index=p_idx
            ))
            
            for i in range(len(text_lines)-1):
                gap = text_lines[i+1].y_center - text_lines[i].y_center
                if gap > 2.0 * avg_height:
                    p_idx += 1
                
                curr_line = text_lines[i+1]
                updated_words = [
                    OCRWord(
                        text=w.text,
                        bbox=w.bbox,
                        center_x=w.center_x,
                        center_y=w.center_y,
                        confidence=w.confidence,
                        word_index=w.word_index,
                        line_index=w.line_index,
                        paragraph_index=p_idx
                    ) for w in curr_line.words
                ]
                updated_lines.append(TextLine(
                    words=updated_words,
                    y_center=curr_line.y_center,
                    bbox=curr_line.bbox,
                    line_index=curr_line.line_index,
                    paragraph_index=p_idx
                ))
            return updated_lines
        else:
            if text_lines:
                single_line = text_lines[0]
                updated_words = [
                    OCRWord(
                        text=w.text,
                        bbox=w.bbox,
                        center_x=w.center_x,
                        center_y=w.center_y,
                        confidence=w.confidence,
                        word_index=w.word_index,
                        line_index=w.line_index,
                        paragraph_index=0
                    ) for w in single_line.words
                ]
                return [TextLine(
                    words=updated_words,
                    y_center=single_line.y_center,
                    bbox=single_line.bbox,
                    line_index=single_line.line_index,
                    paragraph_index=0
                )]
            return []

def get_search_polygon(
    finger: FingerPoint,
    avg_line_height: float,
    avg_word_width: float,
    config: SelectionConfig
) -> np.ndarray:
    """Project the asymmetric search region polygon based on pointing vector or fallback box."""
    x0, y0 = finger.x, finger.y
    
    # Decide direction vector (dx, dy)
    use_direction = (
        config.selection_strategy in (SelectionStrategy.DIRECTION_CONE, SelectionStrategy.POINT, SelectionStrategy.HYBRID) or
        (config.selection_strategy == SelectionStrategy.AUTO and 
         finger.direction is not None and 
         finger.detection_method == "mediapipe")
    )
    
    if use_direction and finger.direction is not None:
        dx, dy = finger.direction
    else:
        dx, dy = 0.0, -1.0  # Point straight up
        
    nx, ny = -dy, dx  # Orthogonal unit vector
    
    if config.selection_strategy in (SelectionStrategy.TOUCH, SelectionStrategy.STATIC_BOX):
        # Touch Mode: use symmetrical above/below search bounds around the finger touch point
        L = 1.0 * avg_line_height
        buffer = 1.0 * avg_line_height
    else:
        L = config.search_height_above * avg_line_height
        buffer = config.search_height_below * avg_line_height
    
    start_hw = avg_word_width / 2.0
    
    # Adaptive cone: If pointing direction confidence is low, widen the search cone to increase tolerance
    conf = max(0.0, min(1.0, finger.confidence))
    confidence_scale = 1.0 + (1.0 - conf) * 1.5  # Up to 2.5x wider search spread when uncertain
    end_hw = (config.search_width_ratio * confidence_scale) * avg_word_width / 2.0
    
    # Trapezoid base and end centers
    bcx, bcy = x0 - buffer * dx, y0 - buffer * dy
    ecx, ecy = x0 + L * dx, y0 + L * dy
    
    # Generate 4-point polygon vertices
    p1 = (bcx + start_hw * nx, bcy + start_hw * ny)
    p2 = (bcx - start_hw * nx, bcy - start_hw * ny)
    p3 = (ecx - end_hw * nx, ecy - end_hw * ny)
    p4 = (ecx + end_hw * nx, ecy + end_hw * ny)
    
    return np.array([p1, p2, p3, p4], dtype=np.float32)

def get_box_distance(x: float, y: float, bbox: Tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance from a point (x, y) to a 2D bounding box (x_min, y_min, x_max, y_max)."""
    x_min, y_min, x_max, y_max = bbox
    dx = max(0.0, x_min - x, x - x_max)
    dy = max(0.0, y_min - y, y - y_max)
    return float(np.hypot(dx, dy))

def select_intended_word(
    finger: FingerPoint,
    ocr_words: List[OCRWord],
    config: SelectionConfig
) -> SelectionResult:
    """Core selection algorithm: projects region, intersects lines, scores candidates, extracts context."""
    # 1. Group words into lines/paragraphs
    text_lines = group_ocr_words(ocr_words, config)
    if not text_lines:
        return SelectionResult(status=SelectionStatus.OCR_EMPTY, finger_point=finger)
        
    # Re-extract all updated words after dynamic clustering
    updated_ocr_words = []
    for line in text_lines:
        updated_ocr_words.extend(line.words)
        
    # 2. Compute average word dimensions
    avg_word_width = np.mean([w.bbox[2] - w.bbox[0] for w in updated_ocr_words])
    avg_line_height = np.mean([w.bbox[3] - w.bbox[1] for w in updated_ocr_words])
    
    # 2.5 Stage 1: Direct Touch Check
    # If the user is touching a word directly (or is within half a line height of it),
    # select that word immediately and bypass directional projection (except for POINT-only modes).
    if config.selection_strategy in (SelectionStrategy.TOUCH, SelectionStrategy.STATIC_BOX, SelectionStrategy.HYBRID, SelectionStrategy.AUTO):
        touch_candidates = []
        for w in updated_ocr_words:
            dist = get_box_distance(finger.x, finger.y, w.bbox)
            if dist <= 0.5 * avg_line_height:
                touch_candidates.append((dist, w))
                
        if touch_candidates:
            touch_candidates.sort(key=lambda item: (
                item[0],
                np.hypot(item[1].center_x - finger.x, item[1].center_y - finger.y)
            ))
            best_touch_word = touch_candidates[0][1]
            min_dist = touch_candidates[0][0]
            
            best_line = None
            for line in text_lines:
                if line.line_index == best_touch_word.line_index:
                    best_line = line
                    break
                    
            if best_line:
                selected_line = " ".join([w.text for w in best_line.words])
                selected_line_words = [w.text for w in best_line.words]
                
                p_lines = [line for line in text_lines if line.paragraph_index == best_line.paragraph_index]
                p_words = []
                for line in p_lines:
                    p_words.extend(line.words)
                selected_paragraph = " ".join([w.text for w in p_words])
                
                sentences = []
                curr_sentence_words = []
                target_sentence_idx = -1
                
                for w in p_words:
                    curr_sentence_words.append(w.text)
                    if w.text and (w.text[-1] in ".!?" or (len(w.text) > 1 and w.text[-2] in ".!?" and w.text[-1] in "\"'")):
                        sentence_str = " ".join(curr_sentence_words)
                        sentences.append(sentence_str)
                        if w.line_index == best_touch_word.line_index and w.word_index == best_touch_word.word_index:
                            target_sentence_idx = len(sentences) - 1
                        curr_sentence_words = []
                        
                if curr_sentence_words:
                    sentence_str = " ".join(curr_sentence_words)
                    sentences.append(sentence_str)
                    for w in p_words[len(p_words) - len(curr_sentence_words):]:
                        if w.line_index == best_touch_word.line_index and w.word_index == best_touch_word.word_index:
                            target_sentence_idx = len(sentences) - 1
                            break
                            
                context = sentences[target_sentence_idx] if target_sentence_idx != -1 else selected_line
                overall_confidence = float(np.clip(finger.confidence * best_touch_word.confidence, 0.0, 1.0))
                status = SelectionStatus.SUCCESS if overall_confidence >= config.confidence_threshold else SelectionStatus.LOW_CONFIDENCE
                
                return SelectionResult(
                    status=status,
                    selected_word=best_touch_word.text,
                    selected_line=selected_line,
                    selected_line_words=selected_line_words,
                    selected_paragraph=selected_paragraph,
                    context=context,
                    confidence=overall_confidence,
                    finger_point=finger,
                    selected_word_bbox=best_touch_word.bbox,
                    word_index=best_touch_word.word_index,
                    line_index=best_touch_word.line_index,
                    paragraph_index=best_touch_word.paragraph_index,
                    candidate_scores=[],
                    selection_reason=f"Direct touch selection: finger is within {min_dist:.1f}px of '{best_touch_word.text}'"
                )
    
    # 3. Project search region polygon
    search_poly = get_search_polygon(finger, avg_line_height, avg_word_width, config)
    
    # 4. Line Intersection Step: Choose best text line
    best_line: Optional[TextLine] = None
    best_line_score = -1.0
    
    use_direction = (
        config.selection_strategy in (SelectionStrategy.DIRECTION_CONE, SelectionStrategy.POINT, SelectionStrategy.HYBRID) or
        (config.selection_strategy == SelectionStrategy.AUTO and 
         finger.direction is not None and 
         finger.detection_method == "mediapipe")
    )
    dx, dy = finger.direction if (use_direction and finger.direction) else (0.0, -1.0)
    
    for line in text_lines:
        # Count words inside search polygon
        words_in_poly = 0
        for w in line.words:
            dist = cv2.pointPolygonTest(search_poly, (w.center_x, w.center_y), False)
            if dist >= 0:
                words_in_poly += 1
                
        # Check if pointing ray intersects this line
        ray_intersects = False
        if use_direction and abs(dy) > 1e-3:
            t = (line.y_center - finger.y) / dy
            buffer_t = config.search_height_below * avg_line_height
            if -buffer_t <= t <= config.search_height_above * avg_line_height:
                rx = finger.x + t * dx
                if line.bbox[0] <= rx <= line.bbox[2]:
                    ray_intersects = True
                    
        # Y proximity score
        y_dist = abs(line.y_center - finger.y)
        y_score = np.exp(-y_dist / (avg_line_height * 2.0))
        
        # Vertical bias (only apply if using directional mode)
        if use_direction:
            is_correct_direction = (line.y_center < finger.y) if dy < 0 else (line.y_center > finger.y)
            dir_bias = 1.0 if is_correct_direction else 0.2
        else:
            dir_bias = 1.0
            
        # Combined line score
        line_score = words_in_poly * 10.0 + (5.0 if ray_intersects else 0.0) + (y_score * 3.0 * dir_bias)
        
        if line_score > best_line_score:
            best_line_score = line_score
            best_line = line
            
    if not best_line or best_line_score < 0.1:
        return SelectionResult(
            status=SelectionStatus.NO_WORD_FOUND,
            finger_point=finger,
            selection_reason="No text line intersected with search region"
        )
        
    # 5. Word Scoring Step: Score all words on the selected line
    scored_candidates: List[ScoredCandidate] = []
    
    # Calculate weights based on chosen SelectionStrategy
    strategy = config.selection_strategy
    if strategy in (SelectionStrategy.TOUCH, SelectionStrategy.STATIC_BOX):
        # Touch Mode: Overlap/proximity dominates, direction/vertical weight is zeroed
        v_weight = 0.0
        h_weight = 0.20
        d_weight = 0.0
        o_weight = 0.80
    elif strategy in (SelectionStrategy.POINT, SelectionStrategy.DIRECTION_CONE):
        # Point Mode: Pointing direction/vertical alignment dominates, touch overlap is minimized
        v_weight = 0.40
        h_weight = 0.10
        d_weight = 0.50
        o_weight = 0.00
    else:
        # Hybrid/Auto Mode: Use a balanced combination of all parameters (config-driven)
        if finger.direction is None:
            v_weight = config.vertical_bias + config.direction_weight / 2.0
            h_weight = config.horizontal_weight + config.direction_weight / 2.0
            d_weight = 0.0
            o_weight = config.overlap_weight
        else:
            v_weight = config.vertical_bias
            h_weight = config.horizontal_weight
            d_weight = config.direction_weight
            o_weight = config.overlap_weight
        
    nx, ny = -dy, dx
    
    for w in best_line.words:
        # Distance vectors relative to finger
        wx, wy = w.center_x - finger.x, w.center_y - finger.y
        d_parallel = wx * dx + wy * dy
        d_perp = wx * nx + wy * ny
        
        # A. Vertical score: Prefer words along the pointing vector
        if d_parallel > 0:
            vertical_score = 1.0
        else:
            vertical_score = float(np.exp(-abs(d_parallel) / avg_line_height))
            
        # B. Horizontal score: Distance perpendicular to pointing vector
        horizontal_score = float(np.exp(-abs(d_perp) / (avg_word_width * 1.5)))
        
        # C. Direction score: Angular alignment
        v_norm = np.hypot(wx, wy)
        if v_norm > 1e-3:
            cos_theta = (wx * dx + wy * dy) / v_norm
            direction_score = float(max(0.0, cos_theta) ** 2) if finger.direction is not None else 0.0
        else:
            direction_score = 1.0 if finger.direction is not None else 0.0
            
        # D. Overlap score: Physical containment
        if w.bbox[0] <= finger.x <= w.bbox[2] and w.bbox[1] <= finger.y <= w.bbox[3]:
            overlap_score = 1.0
        else:
            dx_box = max(0.0, w.bbox[0] - finger.x, finger.x - w.bbox[2])
            dy_box = max(0.0, w.bbox[1] - finger.y, finger.y - w.bbox[3])
            box_dist = np.hypot(dx_box, dy_box)
            overlap_score = float(np.exp(-box_dist / (avg_line_height * 0.5)))
            
        total_score = (
            v_weight * vertical_score +
            h_weight * horizontal_score +
            d_weight * direction_score +
            o_weight * overlap_score
        )
        
        reason = (
            f"V-score: {vertical_score:.2f}, H-score: {horizontal_score:.2f}, "
            f"Dir-score: {direction_score:.2f}, Overlap: {overlap_score:.2f}"
        )
        
        scored_candidates.append(ScoredCandidate(
            word=w,
            vertical_score=vertical_score,
            horizontal_score=horizontal_score,
            direction_score=direction_score,
            overlap_score=overlap_score,
            total_score=total_score,
            scoring_reason=reason
        ))
        
    # Sort candidates by score descending
    scored_candidates = sorted(scored_candidates, key=lambda c: c.total_score, reverse=True)
    if not scored_candidates:
        return SelectionResult(
            status=SelectionStatus.NO_WORD_FOUND,
            finger_point=finger,
            selection_reason="No candidate words scored on chosen line"
        )
        
    best_candidate = scored_candidates[0]
    best_word = best_candidate.word
    
    # 6. Context Reconstruction Step
    selected_line = " ".join([w.text for w in best_line.words])
    selected_line_words = [w.text for w in best_line.words]
    
    # Find paragraph block using lines with same paragraph_index
    p_lines = [line for line in text_lines if line.paragraph_index == best_line.paragraph_index]
    p_words = []
    for line in p_lines:
        p_words.extend(line.words)
    selected_paragraph = " ".join([w.text for w in p_words])
    
    # Sentence grouping algorithm
    sentences = []
    curr_sentence_words = []
    target_sentence_idx = -1
    
    for w in p_words:
        curr_sentence_words.append(w.text)
        # Detect end of sentence (. ! ? or similar punctuation)
        if w.text and (w.text[-1] in ".!?" or (len(w.text) > 1 and w.text[-2] in ".!?" and w.text[-1] in "\"'")):
            sentence_str = " ".join(curr_sentence_words)
            sentences.append(sentence_str)
            if w.line_index == best_word.line_index and w.word_index == best_word.word_index:
                target_sentence_idx = len(sentences) - 1
            curr_sentence_words = []
            
    if curr_sentence_words:
        sentence_str = " ".join(curr_sentence_words)
        sentences.append(sentence_str)
        # Check if the target word is inside this last sentence
        for w in p_words[len(p_words) - len(curr_sentence_words):]:
            if w.line_index == best_word.line_index and w.word_index == best_word.word_index:
                target_sentence_idx = len(sentences) - 1
                break
                
    context = sentences[target_sentence_idx] if target_sentence_idx != -1 else selected_line
    
    # 7. Confidence & SelectionStatus Math
    second_best_score = scored_candidates[1].total_score if len(scored_candidates) > 1 else 0.0
    separation_factor = 1.0 - (second_best_score / best_candidate.total_score) if best_candidate.total_score > 0 else 0.0
    separation_weight = 0.6 + 0.4 * separation_factor

    overall_confidence = float(np.clip(
        best_candidate.total_score * finger.confidence * best_word.confidence * separation_weight,
        0.0, 1.0
    ))
    
    # Check if confidence threshold is met
    status = SelectionStatus.SUCCESS
    if overall_confidence < config.confidence_threshold:
        status = SelectionStatus.LOW_CONFIDENCE
        
    reason_summary = (
        f"Selected '{best_word.text}' (score={best_candidate.total_score:.2f}): "
        f"Separation={separation_factor:.2f}. "
        f"Method={finger.detection_method} (finger_conf={finger.confidence:.2f})."
    )
    
    return SelectionResult(
        status=status,
        selected_word=best_word.text,
        selected_line=selected_line,
        selected_line_words=selected_line_words,
        selected_paragraph=selected_paragraph,
        context=context,
        confidence=overall_confidence,
        finger_point=finger,
        selected_word_bbox=best_word.bbox,
        word_index=best_word.word_index,
        line_index=best_word.line_index,
        paragraph_index=best_word.paragraph_index,
        candidate_scores=scored_candidates[:5],  # Top 5 scored words for debugging
        selection_reason=reason_summary
    )
