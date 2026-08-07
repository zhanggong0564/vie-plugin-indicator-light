from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.common import AICameraModel, VisualReferenceParams
from schemas.data_base import MoMResult


@dataclass
class IndicatorResult(MoMResult):
    """指示灯结果附带场景专属的回流分类。"""

    backflow_category: Optional[str] = None

    def to_dict(self):
        result = super().to_dict()
        if self.backflow_category is not None:
            result["backflow_category"] = self.backflow_category
        return result


class ModelParams(VisualReferenceParams):
    """modelParams 整体模型（guide_line/example_images 设为可选）。"""

    type: int = Field(..., description="产品型号(用于在 AICameraModel 中匹配注册参考图的 Version)")
    # JSON 契约保持 "register"；用 alias 避免与 pydantic BaseModel 属性同名告警。
    register_mode: Optional[bool] = Field(default=None, alias="register", description="注册图是否发生变化")


AICameraModels = AICameraModel


class IndicatorRequest(BaseModel):
    """请求中 json_data 对应的结构化模型。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
    AICameraModel: Optional[List[AICameraModels]] = Field(default_factory=list, description="AICamera模型列表")
