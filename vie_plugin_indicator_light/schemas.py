from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import AICameraModel, VisualReferenceParams


class ModelParams(VisualReferenceParams):
    """modelParams 整体模型（guide_line/example_images 设为可选）。"""

    type: int = Field(..., description="产品型号(用于在 AICameraModel 中匹配注册参考图的 Version)")
    # JSON 契约保持 "register"；用 alias 避免与 pydantic BaseModel 属性同名告警。
    # 注：注册模式当前未在业务层启用（旧 detect 亦未接通），仅保留契约字段。
    register_mode: Optional[bool] = Field(default=None, alias="register", description="是否为注册模式")


AICameraModels = AICameraModel


class IndicatorRequest(BaseModel):
    """请求中 json_data 对应的结构化模型。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
    AICameraModel: Optional[List[AICameraModels]] = Field(default_factory=list, description="AICamera模型列表")
