"""entry_point 模块：导入 business_logic 触发工厂注册，并暴露 indicator_router。"""

import re
from pathlib import Path

import numpy as np

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness
from schemas.exceptions import InvalidParamsError
from .schemas import IndicatorRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("indicator_light")
from .config import IndicatorLightConfig
from .download import download_image
from utils.async_utils import run_sync


class IndicatorRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return IndicatorRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        """数据回流按【物料号】分目录：取请求顶层 ``type``（如 A0SW2163）。

        注意区分两个同名 ``type``：
          - 顶层 ``type``（物料号，如 A0SW2163）：产品唯一编码，是样本归集的正确键 → 作目录名。
          - ``modelParams.type``（int 1/2）：仅用于在 AICameraModel 中按 Version 匹配注册参考图，
            是版本选择器，所有产品都收敛到 1/2，对按产品归集样本无意义。
        顶层 type 缺失时返回 None，框架回退到 _unknown_model 目录。
        """
        t = getattr(request_params, "type", None)
        return str(t).strip() if t else None

    def resolve_backflow_target(self, original_filename, fallback_product_type=None):
        """指示灯专属：沿用框架的场景/型号目录推导（型号=物料号，见 _extract_product_type），
        仅把落盘文件名改为原图末尾时间戳（如 '风电-整机组装1-231-1782460558709.jpg'
        → '1782460558709'）。取不到时间戳则保留框架默认（原图名去扩展名）。

        最终落盘路径：
            data/indicator_light/{YYYY-MM-DD}/{物料号}/{ok|ng}/images|records/{时间戳}.{ext|json}
        """
        target = super().resolve_backflow_target(original_filename, fallback_product_type)
        timestamp = self._extract_timestamp(original_filename)
        if timestamp:
            target.save_stem = timestamp
        return target

    @staticmethod
    def _extract_timestamp(filename):
        """从原图名取末尾时间戳：纯数字名直接用；否则取最后一段 '-<digits>'。无则 None。"""
        stem = Path(filename).stem
        if stem.isdigit():
            return stem
        m = re.search(r"-(\d+)$", stem)
        return m.group(1) if m else None

    async def get_inputs(self, request_params: IndicatorRequest, image: np.ndarray):
        # 注册参考图：在 AICameraModel 中按 Version == modelParams.type 匹配 ModelFile 并下载
        type_ = request_params.modelParams.type
        registered_image_file = ""
        for model in request_params.AICameraModel or []:
            if model.Version != type_:
                continue
            registered_image_file = model.ModelFile
        if not registered_image_file:
            raise InvalidParamsError(f"未找到型号 {type_} 对应的注册参考图")
        cfg = IndicatorLightConfig()
        registered_image = await run_sync(
            download_image,
            registered_image_file,
            max_bytes=cfg.MAX_REGISTERED_IMAGE_MB * 1024 * 1024,
            timeout=(cfg.DOWNLOAD_CONNECT_TIMEOUT, cfg.DOWNLOAD_READ_TIMEOUT),
            allowed_hosts=cfg.ALLOWED_HOSTS,
        )
        return InputParamsBusiness(image=image, registered=registered_image, product_type=str(type_))


indicator_router = IndicatorRouter(
    router_name="indicator_router",
    api_path="/indicator_light_detect",
    summary="指示灯检测接口",
    description="根据输入的图像和注册参考图，比对指示灯状态并返回检测结果",
    detector_type="indicator_light",
    tag="指示灯检测",
)
