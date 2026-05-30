class IndicatorLightConfig:
    """指示灯场景配置（原 config/indicator_light_config.py，迁入插件自描述）。"""

    class ModelPath:
        det_model_path: str = "./weights/IndicatorLightDet_v2.onnx"
        rec_model_path: str = "./weights/IndicatorLightRec_v2.onnx"

    class ConfThreshold:
        det: float = 0.25
        rec: float = 0.25

    JSON_PATH: str = "weights/jsons/standard_embeddings.json"
    SIM_THR: float = 0.7
    IS_CACHE: bool = True
