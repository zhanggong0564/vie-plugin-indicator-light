"""指示灯检测业务逻辑：当前图 vs 注册参考图的 embedding 比对，收敛到模板方法。

旧框架重写了 detect() 做双输入推理；新框架改为：模板已对 ctx.image 推理（ctx.raw_result），
business_post_process 内对 ctx.registered（注册参考图）第二次推理并逐 roi 比对 embedding。
坐标输出像素值，归一化交基类 normalize_hook（数量不匹配的占位项 coordinate 为空，hook 跳过）。
对 malformed 推理结果（None payload / 缺失 embeddings / 缺失 boxes）统一抛 ModelInferenceError。
"""

import numpy as np

from services.api import detection_factory
from services.base import BusinessLogicBase
from schemas.data_base import IndicatorLightEmbedding, MoMResult, DetectionItem
from schemas.exceptions import ModelInferenceError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .indicator_light_det import IndicatorLightDetRec


@detection_factory.register("indicator_light")
class IndicatorLightBusinessAPI(BusinessLogicBase):
    def _initialize_model(self, settings):
        from .config import IndicatorLightConfig
        cfg = IndicatorLightConfig()
        self.sim_thr = cfg.SIM_THR
        try:
            self.detector = IndicatorLightDetRec(
                cfg.ModelPath.det_model_path,
                cfg.ModelPath.rec_model_path,
                cfg.ConfThreshold.det,
            )
        except Exception as e:
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise ModelInferenceError("indicator_light 模型加载失败", scenario="indicator_light", original_error=e)

    def business_post_process(self, ctx: InferenceContext) -> None:
        result = ctx.raw_result  # 当前图 IndicatorLightEmbedding（模板已对 ctx.image 推理）
        registered = ctx.registered
        if registered is None or len(registered) == 0:
            ctx.result = MoMResult(status=False, error_msg="缺少注册参考图(registered)", message="失败")
            return
        result_registered = self.detector.infer(registered)
        ctx.result = self._compare(result, result_registered)

    @staticmethod
    def _embedding_array(value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ModelInferenceError(f"indicator_light {label} 特征维度非法", scenario="indicator_light")
        if float(np.linalg.norm(arr)) == 0.0:
            raise ModelInferenceError(f"indicator_light {label} 特征为空向量", scenario="indicator_light")
        return arr

    @staticmethod
    def _validate_embedding_shapes(std: np.ndarray, cur: np.ndarray) -> None:
        if std.shape != cur.shape:
            raise ModelInferenceError(
                f"indicator_light 特征维度不一致: {std.shape}!={cur.shape}",
                scenario="indicator_light",
            )

    def _compare(
        self, results: IndicatorLightEmbedding, result_registered: IndicatorLightEmbedding
    ) -> MoMResult:
        if result_registered is None:
            raise ModelInferenceError("indicator_light 缺少注册参考图推理结果", scenario="indicator_light")
        if results is None:
            raise ModelInferenceError("indicator_light 缺少当前图推理结果", scenario="indicator_light")

        standard_embeddings = result_registered.embeddings
        if standard_embeddings is None:
            raise ModelInferenceError("indicator_light 未找到注册标准特征", scenario="indicator_light")
        current_embeddings = results.embeddings
        if current_embeddings is None:
            raise ModelInferenceError("indicator_light 未找到当前图特征", scenario="indicator_light")
        if len(standard_embeddings) != len(current_embeddings):
            vision_logger.warning("检测到的指示灯数量与注册的标准特征数量不匹配，可能导致比对结果异常")
            return MoMResult(
                status=False,
                error_msg=f"Number of ROIs does not match the registered standard image "
                          f"{len(standard_embeddings)}!={len(current_embeddings)}.",
                message="失败",
                detailList=[DetectionItem(status=False, scene="", coordinate=[], accuracy=0.0)],
            )
        if results.boxes is None:
            raise ModelInferenceError("indicator_light boxes 为空", scenario="indicator_light")
        if len(results.boxes) < len(current_embeddings):
            raise ModelInferenceError(
                "indicator_light boxes 数量少于 embeddings 数量",
                scenario="indicator_light",
            )
        flag_status = True
        detect_results = MoMResult()
        for i, (std_embedding, embedding) in enumerate(zip(standard_embeddings, current_embeddings)):
            detectionitem = self.compare_embedding(std_embedding, embedding)
            detectionitem.coordinate = results.boxes[i][:4]  # 像素 xyxy，归一化交 normalize_hook
            detect_results.detailList.append(detectionitem)
            if detectionitem.status is False:
                flag_status = False
        detect_results.status = flag_status
        detect_results.message = "success" if flag_status else "failed"
        return detect_results

    def compare_embedding(self, std_embeddings, embeddings) -> DetectionItem:
        std_embeddings = self._embedding_array(std_embeddings, "标准")
        embeddings = self._embedding_array(embeddings, "当前")
        self._validate_embedding_shapes(std_embeddings, embeddings)
        distance = np.dot(std_embeddings, embeddings.T) / (
            np.linalg.norm(std_embeddings) * np.linalg.norm(embeddings)
        )
        distance = (distance + 1) / 2  # cosine [-1,1] -> [0,1]
        return DetectionItem(
            status=bool(distance > self.sim_thr),
            scene="roi",
            accuracy=round(float(distance), 3),
        )
