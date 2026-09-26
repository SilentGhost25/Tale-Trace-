"""
Central config loader. Reads .env once; every other module imports `settings`
from here instead of touching os.environ directly.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    if val:
        v = val.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        return v
    return default


@dataclass(frozen=True)
class Settings:
    app_name: str = _get("APP_NAME", "TaleTrace API")
    debug: bool = _get("DEBUG", "false").lower() == "true"
    log_level: str = _get("LOG_LEVEL", "INFO")

    ocr_space_api_key: str = _get("OCR_SPACE_API_KEY")
    tesseract_cmd: str = _get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ocr_capture_interval: float = float(_get("OCR_CAPTURE_INTERVAL", "6.0"))

    # Unified Groq key or individual keys
    groq_api_key: str = _get("GROQ_API_KEY")
    groq_key_merge: str = _get("GROQ_API_KEY_1") or _get("GROQ_API_KEY")
    groq_key_ai: str = _get("GROQ_API_KEY_2") or _get("GROQ_API_KEY")
    groq_key_learning: str = _get("GROQ_API_KEY_3") or _get("GROQ_API_KEY")

    groq_model: str = _get("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_fast_model: str = _get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

    freesound_api_key: str = _get("FREESOUND_API_KEY")

    esp32_buttons_url: str = _get("ESP32_BUTTONS_URL", "http://192.168.1.26:8080/buttons")
    esp32_display_url: str = _get("ESP32_DISPLAY_URL", "http://192.168.1.26:8080/display")
    esp32_serial_port: str = _get("ESP32_SERIAL_PORT", "")  # Empty string = auto-detect
    esp32_baud_rate: int = int(_get("ESP32_BAUD_RATE", "115200"))

    audio_provider: str = _get("AUDIO_PROVIDER", "offline")
    edge_tts_voice: str = _get("EDGE_TTS_VOICE", "en-US-AriaNeural")

    webcam_device_index: int = int(_get("WEBCAM_DEVICE_INDEX", "0"))
    webcam_width: int = int(_get("WEBCAM_WIDTH", "1280"))
    webcam_height: int = int(_get("WEBCAM_HEIGHT", "720"))

    book_title: str = _get("BOOK_TITLE", "Septopus: Trouble on the High Cs")
    book_author: str = _get("BOOK_AUTHOR", "Jyotin Goel")


settings = Settings()


def require(*keys: str) -> None:
    """Raise a clear error early if a feature's required keys are missing,
    instead of failing deep inside a request with a confusing traceback."""
    missing = [k for k in keys if not getattr(settings, k, "")]
    if missing:
        raise RuntimeError(
            f"Missing required config: {', '.join(missing)}. Check your .env."
        )
