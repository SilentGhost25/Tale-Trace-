"""
Webcam stream and image preprocessing module for Logitech C270 HD WebCam.

- Uses cv2.CAP_DSHOW on Windows for low-latency capture.
- Forces native 1280x720 resolution.
- Includes CLAHE contrast enhancement and Laplacian sharpening from OCRandGESTURE.
- Includes motion stability detection to prevent capturing during page flips.
"""
import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("taletrace.webcam")


def preprocess_image(frame: np.ndarray) -> np.ndarray:
    """
    Applies brightness/contrast enhancement (CLAHE) and Laplacian sharpening.
    Returns single-channel sharpened grayscale image suitable for OCR.
    """
    if frame is None or frame.size == 0:
        return frame

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3x3 Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


class WebcamStream:
    def __init__(
        self,
        device_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps_limit: float = 15.0,
    ):
        self.device_index = device_index if device_index is not None else settings.webcam_device_index
        self.width = width if width is not None else settings.webcam_width
        self.height = height if height is not None else settings.webcam_height
        self.fps_limit = fps_limit

        self._cap = None
        self._latest_frame = None
        self._prev_gray = None
        self._is_stable = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        # On Windows, DirectShow provides faster initialization and proper resolution control
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, backend)

        if not self._cap.isOpened():
            # Fallback to default backend if DSHOW fails
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at device index {self.device_index}. "
                "Check WEBCAM_DEVICE_INDEX in .env and verify no other app is using it."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            "Logitech C270 initialized on index %d: resolution %dx%d (target %dx%d)",
            self.device_index,
            actual_w,
            actual_h,
            self.width,
            self.height,
        )

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        min_interval = 1.0 / self.fps_limit
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = self._cap.read()
            if ok and frame is not None:
                # Check frame stability via frame differencing
                small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
                with self._lock:
                    self._latest_frame = frame
                    if self._prev_gray is not None:
                        diff = cv2.absdiff(small_gray, self._prev_gray)
                        motion_metric = np.mean(diff)
                        self._is_stable = motion_metric < 12.0
                    else:
                        self._is_stable = True
                    self._prev_gray = small_gray
            else:
                log.warning("Webcam frame read failed; retrying in 0.2s.")
                time.sleep(0.2)
                continue

            elapsed = time.time() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_preprocessed_frame(self) -> Optional[np.ndarray]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        return preprocess_image(frame)

    def is_frame_stable(self) -> bool:
        with self._lock:
            return self._is_stable

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        log.info("Webcam stream stopped.")
