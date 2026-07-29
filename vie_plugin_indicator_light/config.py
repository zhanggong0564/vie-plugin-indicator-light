"""Typed runtime configuration for the indicator-light scene."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class IndicatorLightConfig(SceneSettings):
    model_config = SettingsConfigDict(
        env_prefix="INDICATOR_",
        env_file=".env",
        extra="ignore",
    )

    det_model_path: str = "./weights/indicator_light/det_yolo_v2.onnx"
    rec_model_path: str = "./weights/indicator_light/rec_v3.onnx"
    det_conf_threshold: float = Field(default=0.25, ge=0, le=1)
    rec_conf_threshold: float = Field(default=0.25, ge=0, le=1)
    json_path: str = "weights/indicator_light/standard_embeddings.json"
    sim_threshold: float = Field(default=0.7, ge=0, le=1)
    cache_enabled: bool = True
    vector_cache_enabled: bool = True
    vector_cache_path: str = "./data/indicator_light/chroma"
    vector_collection: str = "registered_embeddings"
    download_connect_timeout: float = Field(default=3, gt=0)
    download_read_timeout: float = Field(default=10, gt=0)
    max_registered_image_mb: int = Field(default=20, gt=0)
    allowed_hosts: str = ""

    @property
    def allowed_host_values(self) -> tuple[str, ...]:
        return tuple(
            part.strip().lower()
            for part in self.allowed_hosts.split(",")
            if part.strip()
        )
