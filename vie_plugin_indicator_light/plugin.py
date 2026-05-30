"""entry_point 模块：导入 business_logic 触发工厂注册，并暴露 indicator_router。"""

import cv2
import numpy as np
import requests

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness
from .schemas import IndicatorRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("indicator_light")


class IndicatorRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return IndicatorRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        # 本场景型号字段名为 modelParams.type(int)，重写基类默认提取使数据回流按型号分目录。
        model_params = getattr(request_params, "modelParams", None)
        t = getattr(model_params, "type", None) if model_params else None
        return str(t) if t is not None else None

    def get_inputs(self, request_params: IndicatorRequest, image: np.ndarray):
        # 注册参考图：在 AICameraModel 中按 Version == modelParams.type 匹配 ModelFile 并下载
        type_ = request_params.modelParams.type
        registered_image_file = ""
        for model in request_params.AICameraModel or []:
            if model.Version != type_:
                continue
            registered_image_file = model.ModelFile
        response = requests.get(registered_image_file)
        image_data = np.frombuffer(response.content, np.uint8)
        registered_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return InputParamsBusiness(image=image, registered=registered_image, product_type=str(type_))


indicator_router = IndicatorRouter(
    router_name="indicator_router",
    api_path="/indicator_light_detect",
    summary="指示灯检测接口",
    description="根据输入的图像和注册参考图，比对指示灯状态并返回检测结果",
    detector_type="indicator_light",
    tag="指示灯检测",
)
