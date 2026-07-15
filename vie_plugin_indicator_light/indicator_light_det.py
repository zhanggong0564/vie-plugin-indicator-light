"""指示灯检测器：YOLO 检测 roi + 分类模型出 embedding，组合成 IndicatorLightDetRec。

适配无状态 BaseOnnxInfer：preprocess→(tensor, PreprocMeta)，post_process(outputs, meta)。
"""

from typing import Sequence, Tuple

import cv2
import numpy as np

from services.yolo import YoloOnnxInfer
from services.base import BaseOnnxInfer
from services.base.inference_runner import InferenceRunner
from services.utils import sort_boxes
from schemas.data_base import IndicatorLightEmbedding
from schemas.exceptions import ModelInferenceError
from schemas.inference_context import PreprocMeta


class IndicatorLightDet(YoloOnnxInfer):
    """指示灯 roi 检测（单类 det）。"""

    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, nc=1, confThreshold=confThreshold, nmsThreshold=nmsThreshold, task=task)
        self.id2name = {0: "roi"}


class IndicatorLightRecognition(BaseOnnxInfer):
    """指示灯分类/特征模型：输出 roi 的 embedding 向量。"""

    def __init__(
        self,
        model_path: str,
        img_size: Tuple[int, int] = (224, 224),
        providers=None,
        runner: InferenceRunner | None = None,
    ):
        if runner is not None:
            self._validate_model_metadata(runner)
        super().__init__(model_path, providers=providers, runner=runner)
        self._validate_model_metadata(self.runner)
        self.embedding_dim = self.runner.output_infos[0].shape[1]
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _validate_model_metadata(runner: InferenceRunner) -> None:
        if len(runner.input_infos) != 1 or len(runner.output_infos) != 1:
            raise ValueError("recognition model must have one input and one output")
        input_shape = runner.input_infos[0].shape
        output_shape = runner.output_infos[0].shape
        if not input_shape or isinstance(input_shape[0], int):
            raise ValueError("recognition model input must use dynamic batch")
        if not output_shape or isinstance(output_shape[0], int):
            raise ValueError("recognition model output must use dynamic batch")
        if (
            len(output_shape) < 2
            or not isinstance(output_shape[1], int)
            or output_shape[1] <= 0
        ):
            raise ValueError("recognition model embedding dimension must be positive")

    def _preprocess_image(self, im: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img

    def preprocess(self, im: np.ndarray):
        img = np.expand_dims(self._preprocess_image(im), axis=0)  # -> NCHW
        # 分类模型无缩放语义，构造占位 meta 满足无状态推理链路约定
        meta = PreprocMeta(r=1.0, dw=0.0, dh=0.0, src_shape=im.shape)
        return img, meta

    def infer_batch(self, rois: Sequence[np.ndarray]) -> np.ndarray:
        if not rois:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        tensor = np.stack([self._preprocess_image(roi) for roi in rois])
        outputs = self.runner.run({self.input_names[0]: tensor})
        return self._validate_embeddings(outputs, len(rois))

    def _validate_embeddings(self, outputs, expected_batch: int) -> np.ndarray:
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise ModelInferenceError(
                "recognition model must return exactly one output array"
            )
        embeddings = outputs[0]
        if embeddings.ndim != 2:
            raise ModelInferenceError("recognition embedding output must have rank 2")
        expected_shape = (expected_batch, self.embedding_dim)
        if embeddings.shape != expected_shape:
            raise ModelInferenceError(
                f"recognition embedding output shape must be {expected_shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ModelInferenceError(
                "recognition embedding output must contain only finite values"
            )
        return embeddings

    def post_process(self, output_data, meta) -> np.ndarray:
        return output_data[0]


class IndicatorLightDetRec:
    """检测 + 识别组合：det 找 roi → 按 x 排序 → 逐 roi 出 embedding。"""

    def __init__(self, det_model_path, rec_model_path, confThreshold=0.5, nmsThreshold=0.5):
        self.det = IndicatorLightDet(det_model_path, confThreshold, nmsThreshold)
        self.rec = IndicatorLightRecognition(rec_model_path)

    def infer(self, image: np.ndarray) -> IndicatorLightEmbedding:
        det_result = self.det.infer(image)
        h, w, _ = image.shape
        boxes = []
        for box, score in zip(det_result.boxes, det_result.scores):
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2, score])
        sorted_boxes = np.array(sort_boxes(boxes)[0])  # 按 x1 排序
        embeddings = []
        for box in sorted_boxes:
            x_min, y_min, x_max, y_max, score = box
            roi = image[
                max(int(y_min - 10), 0): min(int(y_max + 10), h),
                max(int(x_min - 10), 0): min(int(x_max + 10), w),
            ]
            embedding = self.rec.infer(roi)
            embeddings.append(embedding.tolist())
        return IndicatorLightEmbedding(
            embeddings=embeddings,
            boxes=sorted_boxes[:, :4].tolist() if len(embeddings) > 0 else [],
            scores=sorted_boxes[:, 4].tolist() if len(embeddings) > 0 else [],
        )
