"""Entry point: register the scene and expose ``indicator_router``."""

import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from routers.backflow_service import BackflowService
from routers.base_router import BaseRouter, DATA_DIR
from schemas.data_base import InputParamsBusiness
from schemas.exceptions import InvalidParamsError
from .registration.models import RegistrationDescriptor
from .schemas import IndicatorRequest
from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册


class IndicatorRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)
        self.backflow_service = IndicatorBackflowService(
            self.detector_type,
            self.resolve_backflow_target,
            DATA_DIR,
        )

    def request_schema(self, json_dict):
        return IndicatorRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        """返回“基础物料号/版本”，供数据回流建立两级目录。

        顶层 ``type`` 可为 ``A0SW2163`` 或 ``A0SW2163-1``；尾部 ``-1/-2``
        会被去除。版本始终取实际用于匹配注册图的 ``modelParams.type``。
        """
        material_no = IndicatorRouter._base_material_no(
            getattr(request_params, "type", None)
        )
        model_params = getattr(request_params, "modelParams", None)
        version = getattr(model_params, "type", None)
        if material_no is None or version is None:
            return None
        return f"{material_no}/{version}"

    @staticmethod
    def _base_material_no(value):
        material_no = str(value).strip() if value else ""
        if not material_no:
            return None
        return re.sub(r"-(?:1|2)$", "", material_no)

    def resolve_backflow_target(self, original_filename, fallback_product_type=None):
        """指示灯专属：沿用框架的场景/型号目录推导（型号=物料号，见 _extract_product_type），
        仅把落盘文件名改为原图末尾时间戳（如 '风电-整机组装1-231-1782460558709.jpg'
        → '1782460558709'）。取不到时间戳则保留框架默认（原图名去扩展名）。

        最终落盘路径：
            data/indicator_light/{YYYY-MM-DD}/{物料号}/{版本}/{ok|ng|review|unmatch}/images|records/{时间戳}.{ext|json}
        """
        material_no = fallback_product_type
        version = None
        if fallback_product_type and "/" in fallback_product_type:
            material_no, version = fallback_product_type.rsplit("/", 1)
        target = super().resolve_backflow_target(original_filename, material_no)
        timestamp = self._extract_timestamp(original_filename)
        return replace(
            target,
            save_stem=timestamp or target.save_stem,
            model_subdir=version,
        )

    @staticmethod
    def _extract_timestamp(filename):
        """从原图名取末尾时间戳：纯数字名直接用；否则取最后一段 '-<digits>'。无则 None。"""
        stem = Path(filename).stem
        if stem.isdigit():
            return stem
        m = re.search(r"-(\d+)$", stem)
        return m.group(1) if m else None

    async def get_inputs(self, request_params: IndicatorRequest, image: np.ndarray):
        # 保持原有语义：存在多个相同 Version 时使用最后一条记录。
        type_ = request_params.modelParams.type
        selected_model = None
        for model in request_params.AICameraModel or []:
            if model.Version != type_:
                continue
            selected_model = model
        if selected_model is None or not selected_model.ModelFile:
            raise InvalidParamsError(f"未找到型号 {type_} 对应的注册参考图")

        descriptor = RegistrationDescriptor(
            registration_id=selected_model.Id,
            material_no=self._base_material_no(request_params.type),
            product_name=selected_model.ProductName,
            version=selected_model.Version,
            model_file=selected_model.ModelFile,
            create_time=selected_model.CreateTime,
            update_time=selected_model.UpdateTime,
            register_mode=request_params.modelParams.register_mode,
        )
        return InputParamsBusiness(
            image=image,
            product_type=str(type_),
            extra={"registration": asdict(descriptor)},
        )


class IndicatorBackflowService(BackflowService):
    @staticmethod
    def classify_result(result_dict: dict) -> str:
        if result_dict.get("backflow_category") == "unmatch":
            return "unmatch"
        return BackflowService.classify_result(result_dict)


indicator_router = IndicatorRouter(
    router_name="indicator_router",
    api_path="/indicator_light_detect",
    summary="指示灯检测接口",
    description="根据输入的图像和注册参考图，比对指示灯状态并返回检测结果",
    detector_type="indicator_light",
    tag="指示灯检测",
)
