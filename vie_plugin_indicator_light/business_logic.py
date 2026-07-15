"""指示灯当前图与持久化注册 embedding 的业务比对。"""

import numpy as np

from services.api import detection_factory
from services.base import BusinessLogicBase
from schemas.data_base import IndicatorLightEmbedding, MoMResult, DetectionItem
from schemas.exceptions import ModelInferenceError, VisionAPIError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .config import IndicatorLightConfig
from .download import download_image
from .indicator_light_det import IndicatorLightDetRec
from .registration.models import RegistrationDescriptor, pipeline_fingerprint
from .registration.resolver import RegistrationResolver
from .registration.store import NullRegistrationStore


def _create_chroma_store(path: str, collection: str, fingerprint: str):
    """Import Chroma only when persistent caching is actually enabled."""
    from .registration.chroma_store import ChromaRegistrationStore

    return ChromaRegistrationStore(path, collection, fingerprint)


@detection_factory.register("indicator_light")
class IndicatorLightBusinessAPI(BusinessLogicBase):
    def _initialize_model(self, settings):
        cfg = IndicatorLightConfig()
        self.sim_thr = cfg.SIM_THR
        try:
            self.detector = IndicatorLightDetRec(
                cfg.ModelPath.det_model_path,
                cfg.ModelPath.rec_model_path,
                cfg.ConfThreshold.det,
            )
        except ModelInferenceError as e:
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise ModelInferenceError(
                e.error_msg,
                scenario="indicator_light",
                original_error=e,
            ) from e
        except Exception as e:
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise ModelInferenceError("indicator_light 模型加载失败", scenario="indicator_light", original_error=e)
        self._initialize_registration(cfg)

    def _initialize_registration(self, cfg: IndicatorLightConfig) -> None:
        store = NullRegistrationStore()
        fingerprint = "cache-disabled"
        if cfg.INDICATOR_VECTOR_CACHE_ENABLED:
            try:
                fingerprint = pipeline_fingerprint(
                    cfg.ModelPath.det_model_path,
                    cfg.ModelPath.rec_model_path,
                )
            except Exception as exc:
                vision_logger.error(
                    "indicator_light 模型指纹计算失败: error_type={}",
                    type(exc).__name__,
                )
                raise ModelInferenceError(
                    "indicator_light 模型指纹计算失败",
                    scenario="indicator_light",
                    original_error=exc,
                ) from exc
            try:
                store = _create_chroma_store(
                    cfg.INDICATOR_VECTOR_CACHE_PATH,
                    cfg.INDICATOR_VECTOR_COLLECTION,
                    fingerprint,
                )
            except Exception as exc:
                vision_logger.warning(
                    "indicator_light 注册向量缓存初始化失败，使用无缓存模式: error_type={}",
                    type(exc).__name__,
                )

        self.registration_resolver = RegistrationResolver(
            store=store,
            downloader=download_image,
            infer=self.detector.infer,
            pipeline_fingerprint=fingerprint,
            download_options={
                "max_bytes": cfg.MAX_REGISTERED_IMAGE_MB * 1024 * 1024,
                "timeout": (
                    cfg.DOWNLOAD_CONNECT_TIMEOUT,
                    cfg.DOWNLOAD_READ_TIMEOUT,
                ),
                "allowed_hosts": cfg.ALLOWED_HOSTS,
            },
        )

    def preprocess_hook(self, ctx: InferenceContext) -> None:
        registration_data = ctx.extra.get("registration")
        if registration_data is not None:
            try:
                descriptor = RegistrationDescriptor(**registration_data)
            except (TypeError, ValueError) as exc:
                raise ModelInferenceError(
                    "indicator_light 注册描述参数非法",
                    scenario="indicator_light",
                    original_error=exc,
                ) from exc

            try:
                embeddings = self.registration_resolver.resolve(descriptor)
            except VisionAPIError:
                raise
            except Exception as exc:
                raise ModelInferenceError(
                    "indicator_light 注册向量解析失败",
                    scenario="indicator_light",
                    original_error=exc,
                ) from exc
            ctx.extra = dict(ctx.extra)
            ctx.extra["registered_result"] = IndicatorLightEmbedding(
                embeddings=[list(vector) for vector in embeddings]
            )
            return

        registered = ctx.registered
        if registered is not None and len(registered) > 0:
            ctx.extra = dict(ctx.extra)
            ctx.extra["registered_result"] = self.detector.infer(registered)

    def business_post_process(self, ctx: InferenceContext) -> None:
        result = ctx.raw_result  # 当前图 IndicatorLightEmbedding（模板已对 ctx.image 推理）
        result_registered = ctx.extra.get("registered_result")
        if result_registered is None:
            ctx.result = MoMResult(status=False, error_msg="缺少注册参考图(registered)", message="失败")
            return
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
