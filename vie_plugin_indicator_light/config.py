import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a valid boolean value, got {raw!r}")


def _env_hosts(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


class IndicatorLightConfig:
    """指示灯场景配置（原 config/indicator_light_config.py，迁入插件自描述）。"""

    class ModelPath:
        det_model_path: str = "./weights/indicator_light/det_yolo_v2.onnx"
        rec_model_path: str = "./weights/indicator_light/rec_v3.onnx"

    class ConfThreshold:
        det: float = 0.25
        rec: float = 0.25

    JSON_PATH: str = "weights/indicator_light/standard_embeddings.json"
    SIM_THR: float = 0.7
    IS_CACHE: bool = True
    INDICATOR_VECTOR_CACHE_ENABLED: bool = _env_bool("INDICATOR_VECTOR_CACHE_ENABLED", True)
    INDICATOR_VECTOR_CACHE_PATH: str = os.getenv(
        "INDICATOR_VECTOR_CACHE_PATH", "./data/indicator_light/chroma"
    )
    INDICATOR_VECTOR_COLLECTION: str = os.getenv(
        "INDICATOR_VECTOR_COLLECTION", "registered_embeddings"
    )
    DOWNLOAD_CONNECT_TIMEOUT: float = float(os.getenv("INDICATOR_DOWNLOAD_CONNECT_TIMEOUT", "3"))
    DOWNLOAD_READ_TIMEOUT: float = float(os.getenv("INDICATOR_DOWNLOAD_READ_TIMEOUT", "10"))
    MAX_REGISTERED_IMAGE_MB: int = int(os.getenv("INDICATOR_MAX_REGISTERED_IMAGE_MB", "20"))
    ALLOWED_HOSTS: tuple[str, ...] = _env_hosts("INDICATOR_ALLOWED_HOSTS")
