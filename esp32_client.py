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
    intend: bool = False
    intend_pressed: bool = False
    intend_released: bool = False
    connected: bool = False
    mode: str = ""


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
            conn = serial.Serial()
            conn.port = port_to_try
            conn.baudrate = settings.esp32_baud_rate
            conn.timeout = 0.05
            conn.dtr = False
            conn.rts = False
            conn.open()
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
                # Active poll request: send POLL command so ESP32 responds immediately
                try:
                    conn.write(b"POLL\n")
                    conn.flush()
                except Exception:
                    pass

                found_state = None
                saw_pressed = False
                saw_released = False
                lines_read = 0

                # Limit max lines per poll iteration to prevent infinite loop on buffer backlog
                while conn.in_waiting > 0 and lines_read < 25:
                    lines_read += 1
                    try:
                        line = conn.readline().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        break
                    if "INTENT PRESSED" in line:
                        saw_pressed = True
                    elif "INTENT RELEASED" in line:
                        saw_released = True
                    elif line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            m = bool(data.get("btn_momentary", False))
                            t = bool(data.get("btn_toggle", False))
                            both = bool(data.get("both_active", m and t))
                            i = bool(data.get("intend", False))
                            if bool(data.get("intend_pressed", False)):
                                saw_pressed = True
                            if bool(data.get("intend_released", False)):
                                saw_released = True
                            mode_str = str(data.get("mode", ""))
                            found_state = ButtonState(
                                momentary=m,
                                toggle=t,
                                both=both,
                                intend=i or saw_pressed,
                                intend_pressed=saw_pressed,
                                intend_released=saw_released,
                                connected=True,
                                mode=mode_str,
                            )
                        except json.JSONDecodeError:
                            pass

                # If buffer accumulation was large (>20 lines), clear old overflow
                if conn.in_waiting > 30:
                    try:
                        conn.reset_input_buffer()
                    except Exception:
                        pass

                if saw_pressed and (found_state is None or not found_state.intend_pressed):
                    base_m = found_state.momentary if found_state else (_last_button_state.momentary if _last_button_state else False)
                    base_t = found_state.toggle if found_state else (_last_button_state.toggle if _last_button_state else False)
                    base_both = base_m and base_t
                    found_state = ButtonState(
                        momentary=base_m,
                        toggle=base_t,
                        both=base_both,
                        intend=True,
                        intend_pressed=True,
                        intend_released=False,
                        connected=True,
                    )

                if found_state is not None:
                    # Save steady state with edges cleared for subsequent polls
                    _last_button_state = ButtonState(
                        momentary=found_state.momentary,
                        toggle=found_state.toggle,
                        both=found_state.both,
                        intend=found_state.intend,
                        intend_pressed=False,
                        intend_released=False,
                        connected=True,
                        mode=found_state.mode,
                    )
                    return found_state
        except Exception as e:
            log.debug("Serial port error during button poll: %s", e)
            # Reset connection on serial communication failure so next loop attempts clean reconnect
            with _serial_lock:
                try:
                    if _serial_conn:
                        _serial_conn.close()
                except Exception:
                    pass
                _serial_conn = None

    if _last_button_state is not None:
        return ButtonState(
            momentary=_last_button_state.momentary,
            toggle=_last_button_state.toggle,
            both=_last_button_state.both,
            intend=_last_button_state.intend,
            intend_pressed=False,
            intend_released=False,
            connected=_last_button_state.connected,
            mode=_last_button_state.mode,
        )

    # 2. Fallback to HTTP poll
    if settings.esp32_buttons_url:
        try:
            resp = requests.get(settings.esp32_buttons_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                m = bool(data.get("btn_momentary", False))
                t = bool(data.get("btn_toggle", False))
                both = bool(data.get("both_active", m and t))
                i = bool(data.get("intend", False))
                i_pressed = bool(data.get("intend_pressed", False))
                i_released = bool(data.get("intend_released", False))
                mode_str = str(data.get("mode", ""))
                state = ButtonState(
                    momentary=m,
                    toggle=t,
                    both=both,
                    intend=i,
                    intend_pressed=i_pressed,
                    intend_released=i_released,
                    connected=True,
                    mode=mode_str,
                )
                _last_button_state = state
                return state
        except Exception:
            pass

    return ButtonState(connected=False)


def send_display_text(text: str, timeout: float = 1.5) -> bool:
    """Sends display string to ESP32 OLED via Serial UART or HTTP POST."""
    if not text:
        return False

    # Normalize unicode punctuation and smart quotes to standard ASCII for U8g2
    clean_text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
    )
    # Strip non-ASCII characters (emojis, accents) so U8g2 font renders cleanly without garbled glyphs
    clean_text = clean_text.encode("ascii", errors="ignore").decode("ascii")
    # Preserve newlines by escaping them as \n for serial transfer
    clean_text = clean_text.replace("\r", "").replace("\n", "\\n")

    # 1. Attempt Serial UART transmission
    conn = _get_serial_connection()
    if conn is not None:
        try:
            with _serial_lock:
                payload = f"DISPLAY:{clean_text}\n".encode("utf-8")
                conn.write(payload)
                conn.flush()
                log.info("OLED display updated via Serial: %s", clean_text[:40])
                return True
        except Exception as e:
            log.debug("Serial write error: %s", e)

    # 2. Fallback to HTTP POST
    if settings.esp32_display_url:
        try:
            resp = requests.post(
                settings.esp32_display_url,
                data=clean_text.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            log.debug("HTTP display push failed: %s", e)

    return False
