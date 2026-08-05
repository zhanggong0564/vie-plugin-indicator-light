"""指示灯当前图与持久化注册 embedding 的业务比对。"""

from datetime import datetime
import os

import cv2
import numpy as np

from config import settings
from routers.backflow_service import BackflowService
from routers.upload_persistence import write_bytes_atomically
from services.base import BusinessLogicBase
from services.inference import (
    OnnxRuntimeOptions,
    RunnerSpec,
    create_inference_runner,
)
from services.scenario_registry import scenario_registry
from schemas.data_base import IndicatorLightEmbedding, MoMResult, DetectionItem
from schemas.exceptions import ModelInferenceError, VisionAPIError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .config import IndicatorLightConfig
from .download import download_image
from .indicator_light_det import IndicatorLightDetRec
from .layout_matching import (
    match_layout_fast,
    match_layout_orb,
)
from .registration.models import RegistrationDescriptor, pipeline_fingerprint
from .registration.resolver import RegistrationResolver
from .registration.store import NullRegistrationStore
from .schemas import IndicatorResult


def _create_chroma_store(path: str, collection: str, fingerprint: str):
    """Import Chroma only when persistent caching is actually enabled."""
    from .registration.chroma_store import ChromaRegistrationStore

    return ChromaRegistrationStore(path, collection, fingerprint)


@scenario_registry.register("indicator_light")
class IndicatorLightBusinessAPI(BusinessLogicBase):
    def _initialize_model(self, settings):
        cfg = IndicatorLightConfig()
        self.sim_thr = cfg.sim_threshold
        created_runners = []
        pipeline = None
        try:
            options = OnnxRuntimeOptions.from_settings(settings)
            detection_runner = create_inference_runner(
                RunnerSpec(
                    scenario="indicator_light",
                    onnx_path=cfg.det_model_path,
                ),
                options,
            )
            created_runners.append(detection_runner)
            recognition_runner = create_inference_runner(
                RunnerSpec(
                    scenario="indicator_light",
                    onnx_path=cfg.rec_model_path,
                ),
                options,
            )
            created_runners.append(recognition_runner)
            pipeline = IndicatorLightDetRec(
                detection_runner=detection_runner,
                recognition_runner=recognition_runner,
                confThreshold=cfg.det_conf_threshold,
            )
            self._initialize_registration(cfg, pipeline)
            self.detector = pipeline
        except ModelInferenceError as e:
            self._rollback_initialization(pipeline, created_runners)
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise ModelInferenceError(
                e.error_msg,
                scenario="indicator_light",
                original_error=e,
            ) from e
        except Exception as e:
            self._rollback_initialization(pipeline, created_runners)
            vision_logger.error(f"IndicatorLightBusinessAPI init error: {e}")
            raise ModelInferenceError(
                "indicator_light 模型加载失败",
                scenario="indicator_light",
                original_error=e,
            ) from e

    @staticmethod
    def _rollback_initialization(pipeline, created_runners) -> None:
        resources = [pipeline] if pipeline is not None else created_runners
        for resource in resources:
            try:
                resource.close()
            except Exception as close_error:
                vision_logger.warning(
                    "indicator_light 初始化回滚清理失败: error_type={}",
                    type(close_error).__name__,
                )

    def _initialize_registration(
        self,
        cfg: IndicatorLightConfig,
        pipeline: IndicatorLightDetRec,
    ) -> None:
        store = NullRegistrationStore()
        fingerprint = "cache-disabled"
        if cfg.vector_cache_enabled:
            try:
                fingerprint = pipeline_fingerprint(
                    cfg.det_model_path,
                    cfg.rec_model_path,
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
                    cfg.vector_cache_path,
                    cfg.vector_collection,
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
            infer=pipeline.infer,
            pipeline_fingerprint=fingerprint,
            download_options={
                "max_bytes": cfg.max_registered_image_mb * 1024 * 1024,
                "timeout": (
                    cfg.download_connect_timeout,
                    cfg.download_read_timeout,
                ),
                "allowed_hosts": cfg.allowed_host_values,
            },
            image_callback=self._archive_registered_image,
            image_required=self._registered_image_missing,
        )

    @staticmethod
    def _registered_image_path(descriptor) -> str:
        material_no = BackflowService.sanitize_dir_name(descriptor.material_no)
        registration_id = BackflowService.sanitize_dir_name(descriptor.registration_id)
        return BackflowService.safe_path(
            os.path.abspath(settings.DATA_DIR),
            "indicator_light",
            datetime.now().date().isoformat(),
            material_no,
            "registered",
            f"{registration_id}.jpg",
        )

    @classmethod
    def _registered_image_missing(cls, descriptor) -> bool:
        return not os.path.exists(cls._registered_image_path(descriptor))

    @classmethod
    def _archive_registered_image(cls, descriptor, image: np.ndarray) -> None:
        """按日期和物料号保存实际参与比对的注册图。"""
        target = cls._registered_image_path(descriptor)
        if os.path.exists(target):
            return
        encoded, payload = cv2.imencode(".jpg", image)
        if not encoded:
            raise ValueError("指示灯注册图编码失败")
        write_bytes_atomically(payload.tobytes(), target)

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
                resolved = self.registration_resolver.resolve(descriptor)
            except VisionAPIError:
                raise
            except Exception as exc:
                raise ModelInferenceError(
                    "indicator_light 注册向量解析失败",
                    scenario="indicator_light",
                    original_error=exc,
                ) from exc
            ctx.extra = dict(ctx.extra)
            ctx.extra["registered_result"] = resolved.generation.to_inference_result()
            ctx.extra["registration_descriptor"] = descriptor
            if resolved.image is not None:
                ctx.extra["registered_image"] = resolved.image
            return

        registered = ctx.registered
        if registered is not None and len(registered) > 0:
            ctx.extra = dict(ctx.extra)
            ctx.extra["registered_result"] = self.detector.infer(registered)
            ctx.extra["registered_image"] = registered

    def business_post_process(self, ctx: InferenceContext) -> None:
        result = ctx.raw_result  # 当前图 IndicatorLightEmbedding（模板已对 ctx.image 推理）
        result_registered = ctx.extra.get("registered_result")
        if result_registered is None:
            ctx.result = MoMResult(status=False, error_msg="缺少注册参考图(registered)", message="失败")
            return
        self._validate_result_layout(result, "当前图")
        self._validate_result_layout(result_registered, "注册图")
        layout_match = match_layout_fast(result_registered, result)
        if layout_match is None and len(result.embeddings) >= len(result_registered.embeddings):
            registered_image = ctx.extra.get("registered_image")
            descriptor = ctx.extra.get("registration_descriptor")
            if registered_image is None and descriptor is not None:
                archived_path = self._registered_image_path(descriptor)
                if os.path.exists(archived_path):
                    registered_image = cv2.imread(archived_path)
                    if registered_image is None:
                        vision_logger.warning(
                            "指示灯归档注册图读取失败: registration_id={}",
                            descriptor.registration_id,
                        )
            if registered_image is None and descriptor is not None:
                try:
                    registered_image = self.registration_resolver.download_image(descriptor)
                except Exception as exc:
                    vision_logger.warning(
                        "指示灯异常配准读取注册图失败: registration_id={}, error_type={}",
                        descriptor.registration_id,
                        type(exc).__name__,
                    )
            if registered_image is not None:
                try:
                    layout_match = match_layout_orb(
                        result_registered,
                        result,
                        registered_image,
                        ctx.image,
                    )
                except (cv2.error, TypeError, ValueError) as exc:
                    vision_logger.warning(
                        "指示灯 ORB 配准失败: error_type={}",
                        type(exc).__name__,
                    )
        if layout_match is None:
            ctx.result = self._unmatch_result(
                len(result_registered.embeddings),
                len(result.embeddings),
            )
            return
        if layout_match.extra_current_indices:
            vision_logger.warning(
                "指示灯忽略未匹配候选框: category=unmatched_extra_detection, count={}, method={}",
                len(layout_match.extra_current_indices),
                layout_match.method,
            )
        ctx.result = self._compare(result, result_registered, layout_match.pairs)

    @staticmethod
    def _validate_result_layout(result: IndicatorLightEmbedding, label: str) -> None:
        if result.embeddings is None or result.boxes is None:
            raise ModelInferenceError(
                f"indicator_light {label} embeddings 或 boxes 为空",
                scenario="indicator_light",
            )
        if len(result.embeddings) != len(result.boxes):
            raise ModelInferenceError(
                f"indicator_light {label} boxes 与 embeddings 数量不一致",
                scenario="indicator_light",
            )

    @staticmethod
    def _unmatch_result(registered_count: int, current_count: int) -> IndicatorResult:
        if registered_count == current_count:
            reason = (
                "Indicator counts match, but the layout cannot be matched reliably "
                f"({registered_count})."
            )
        else:
            reason = (
                "Unable to match all registered indicator positions "
                f"{registered_count}!={current_count}."
            )
        return IndicatorResult(
            status=False,
            error_msg=reason,
            message="失败",
            detailList=[
                DetectionItem(status=False, scene="", coordinate=[], accuracy=0.0)
            ],
            backflow_category="unmatch",
        )

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
        self,
        results: IndicatorLightEmbedding,
        result_registered: IndicatorLightEmbedding,
        pairs=None,
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
        if results.boxes is None:
            raise ModelInferenceError("indicator_light boxes 为空", scenario="indicator_light")
        if len(results.boxes) < len(current_embeddings):
            raise ModelInferenceError(
                "indicator_light boxes 数量少于 embeddings 数量",
                scenario="indicator_light",
            )
        flag_status = True
        detect_results = MoMResult()
        if pairs is None:
            pairs = tuple(
                (index, index)
                for index in range(min(len(standard_embeddings), len(current_embeddings)))
            )
        for registered_index, current_index in pairs:
            std_embedding = standard_embeddings[registered_index]
            embedding = current_embeddings[current_index]
            detectionitem = self.compare_embedding(std_embedding, embedding)
            detectionitem.coordinate = results.boxes[current_index][:4]
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
