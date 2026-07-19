"""指示灯检测器：YOLO 检测 roi + 分类模型出 embedding，组合成 IndicatorLightDetRec。

模型保持无每请求状态：preprocess 返回 tensor 和 PreprocMeta，post_process 消费二者。
"""

from typing import Sequence, Tuple

import cv2
import numpy as np

from services.base import BaseVisionInfer
from services.inference import InferenceRunner
from services.vision.boxes import sort_boxes
from services.yolo import YoloInfer
from schemas.data_base import IndicatorLightEmbedding
from schemas.exceptions import ModelInferenceError
from schemas.inference_context import PreprocMeta


class IndicatorLightDet(YoloInfer):
    """指示灯 roi 检测（单类 det）。"""

    def __init__(
        self,
        runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
        task="det",
    ):
        super().__init__(
            nc=1,
            runner=runner,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
            task=task,
        )
        self.id2name = {0: "roi"}


class IndicatorLightRecognition(BaseVisionInfer):
    """指示灯分类/特征模型：输出 roi 的 embedding 向量。"""

    def __init__(
        self,
        *,
        runner: InferenceRunner,
        img_size: Tuple[int, int] = (224, 224),
    ):
        self._validate_model_metadata(runner)
        super().__init__(runner)
        self.embedding_dim = runner.output_infos[0].shape[1]
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _validate_model_metadata(runner: InferenceRunner) -> None:
        if len(runner.input_infos) != 1 or len(runner.output_infos) != 1:
            raise ModelInferenceError(
                "recognition model must have one input and one output"
            )
        input_shape = runner.input_infos[0].shape
        output_shape = runner.output_infos[0].shape
        if not input_shape or isinstance(input_shape[0], int):
            raise ModelInferenceError(
                "recognition model input must use dynamic batch；固定 batch 不支持，"
                "请使用 rec_v3.onnx 或将旧识别模型转换为 rec_v3.onnx"
            )
        if not output_shape or isinstance(output_shape[0], int):
            raise ModelInferenceError(
                "recognition model output must use dynamic batch；固定 batch 不支持，"
                "请使用 rec_v3.onnx 或将旧识别模型转换为 rec_v3.onnx"
            )
        if (
            len(output_shape) < 2
            or not isinstance(output_shape[1], int)
            or output_shape[1] <= 0
        ):
            raise ModelInferenceError(
                "recognition model embedding dimension must be positive"
            )

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
        if (
            not isinstance(outputs, Sequence)
            or len(outputs) != 1
            or not isinstance(outputs[0], np.ndarray)
        ):
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
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise ModelInferenceError(
                "recognition embedding output must use a floating-point dtype"
            )
        if not np.isfinite(embeddings).all():
            raise ModelInferenceError(
                "recognition embedding output must contain only finite values"
            )
        return embeddings

    def post_process(self, output_data, meta) -> np.ndarray:
        return output_data[0]


class IndicatorLightDetRec:
    """检测 + 识别组合：det 找 roi → 按 x 排序 → 批量输出 embedding。"""

    def __init__(
        self,
        *,
        detection_runner: InferenceRunner,
        recognition_runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
    ):
        self.det = IndicatorLightDet(
            detection_runner,
            confThreshold,
            nmsThreshold,
        )
        self.rec = IndicatorLightRecognition(runner=recognition_runner)

    def infer(self, image: np.ndarray) -> IndicatorLightEmbedding:
        det_result = self.det.infer(image)
        h, w, _ = image.shape
        boxes = []
        for box, score in zip(det_result.boxes, det_result.scores):
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2, score])
        if not boxes:
            return IndicatorLightEmbedding()
        sorted_boxes = np.array(sort_boxes(boxes)[0])  # 按 x1 排序
        rois = []
        for box in sorted_boxes:
            x_min, y_min, x_max, y_max, score = box
            roi = image[
                max(int(y_min - 10), 0): min(int(y_max + 10), h),
                max(int(x_min - 10), 0): min(int(x_max + 10), w),
            ]
            if roi.size == 0:
                raise ModelInferenceError("detected ROI is empty after clipping")
            rois.append(roi)
        embedding_array = self.rec.infer_batch(rois)
        return IndicatorLightEmbedding(
            embeddings=embedding_array.tolist(),
            boxes=sorted_boxes[:, :4].tolist(),
            scores=sorted_boxes[:, 4].tolist(),
        )

    def close(self) -> None:
        first_error = None
        for model in (self.det, self.rec):
            try:
                model.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
