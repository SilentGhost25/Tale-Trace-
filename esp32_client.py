"""
Dual Serial / HTTP Client for the ESP32 DevKit Rig (buttons_and_oled.ino).

Supports:
1. Low-latency USB Serial UART (auto-detected or configured in .env).
2. WiFi HTTP endpoints (GET /buttons and POST /display).
3. Graceful degradation: returns disconnected state instead of raising exceptions.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from config import settings

log = logging.getLogger("taletrace.esp32")

_serial_conn: Optional[object] = None
_serial_lock = threading.Lock()
_last_button_state = None


@dataclass
class ButtonState:
    momentary: bool = False
    toggle: bool = False
    both: bool = False
    connected: bool = False


def _get_serial_connection():
    global _serial_conn
    if serial is None:
        return None

    with _serial_lock:
        if _serial_conn is not None and getattr(_serial_conn, "is_open", False):
            return _serial_conn

        # Check configured port or probe available ports
        port_to_try = settings.esp32_serial_port
        if not port_to_try:
            available = list(serial.tools.list_ports.comports())
            for p in available:
                desc = p.description.lower()
                # Prioritize USB serial chips commonly used on ESP32
                if any(x in desc for x in ["cp210", "ch340", "ch910", "usb serial", "uart"]):
                    port_to_try = p.device
                    break

        if not port_to_try:
            return None

        try:
            conn = serial.Serial(port_to_try, baudrate=settings.esp32_baud_rate, timeout=0.1)
            _serial_conn = conn
            log.info("Connected to ESP32 on Serial Port: %s", port_to_try)
            return _serial_conn
        except Exception as e:
            log.debug("Could not open serial port %s: %s", port_to_try, e)
            _serial_conn = None
            return None


def get_button_state(timeout: float = 0.3) -> ButtonState:
    """Polls hardware button states via Serial UART first, falling back to HTTP."""
    global _last_button_state

    # 1. Attempt Serial UART poll
    conn = _get_serial_connection()
    if conn is not None:
        try:
            with _serial_lock:
                # Send poll command if needed or read last available line
                while conn.in_waiting > 0:
                    line = conn.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("{") and line.endswith("}"):
                        data = json.loads(line)
                        m = bool(data.get("btn_momentary", False))
                        t = bool(data.get("btn_toggle", False))
                        both = m and t
                        state = ButtonState(momentary=m, toggle=t, both=both, connected=True)
                        _last_button_state = state
                        return state
            if _last_button_state is not None:
                return _last_button_state
        except Exception as e:
            log.debug("Serial read error: %s", e)

    # 2. Fallback to HTTP poll
    if settings.esp32_buttons_url:
        try:
            resp = requests.get(settings.esp32_buttons_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                m = bool(data.get("btn_momentary", False))
                t = bool(data.get("btn_toggle", False))
                both = bool(data.get("both_active", m and t))
                state = ButtonState(momentary=m, toggle=t, both=both, connected=True)
                _last_button_state = state
                return state
        except Exception:
            pass

    return ButtonState(connected=False)


def send_display_text(text: str, timeout: float = 1.5) -> bool:
    """Sends display string to ESP32 OLED via Serial UART or HTTP POST."""
    if not text:
        return False

    # 1. Attempt Serial UART transmission
    conn = _get_serial_connection()
    if conn is not None:
        try:
            with _serial_lock:
                payload = f"DISPLAY:{text}\n".encode("utf-8")
                conn.write(payload)
                return True
        except Exception as e:
            log.debug("Serial write error: %s", e)

    # 2. Fallback to HTTP POST
    if settings.esp32_display_url:
        try:
            resp = requests.post(
                settings.esp32_display_url,
                data=text.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            log.debug("HTTP display push failed: %s", e)

    return False
