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


def find_logitech_device_index() -> Optional[int]:
    """Finds the device index corresponding specifically to the Logitech webcam."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        for idx, name in enumerate(devices):
            if any(term in name.lower() for term in ["logi", "c270", "logitech"]):
                log.info("Auto-detected Logitech webcam: '%s' at index %d", name, idx)
                return idx
    except Exception as e:
        log.debug("Device enumeration fallback: %s", e)
    return None


class WebcamStream:
    def __init__(
        self,
        device_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps_limit: float = 15.0,
    ):
        detected_idx = find_logitech_device_index()
        if device_index is not None:
            self.device_index = device_index
        elif detected_idx is not None:
            self.device_index = detected_idx
        else:
            self.device_index = settings.webcam_device_index

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
        # Fast-path: try pre-detected or configured device index immediately
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index)

        opened = False
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self._cap = cap
                opened = True

        # Fallback to index probing only if initial direct attempt failed
        if not opened:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for idx in [0, 1]:
                if idx == self.device_index:
                    continue
                for backend in backends:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self._cap = cap
                            self.device_index = idx
                            opened = True
                            break
                        cap.release()
                if opened:
                    break

        if not opened or self._cap is None:
            raise RuntimeError(
                f"Could not open Logitech webcam at device index {self.device_index}. "
                "Verify the camera is plugged into USB and not opened by another application."
            )

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
