"""指示灯检测器：YOLO 检测 roi + 分类模型出 embedding，组合成 IndicatorLightDetRec。

适配无状态 BaseOnnxInfer：preprocess→(tensor, PreprocMeta)，post_process(outputs, meta)。
"""

from typing import Tuple

import cv2
import numpy as np

from services.yolo import YoloOnnxInfer
from services.base import BaseOnnxInfer
from services.utils import sort_boxes
from schemas.data_base import IndicatorLightEmbedding
from schemas.inference_context import PreprocMeta


class IndicatorLightDet(YoloOnnxInfer):
    """指示灯 roi 检测（单类 det）。"""

    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, nc=1, confThreshold=confThreshold, nmsThreshold=nmsThreshold, task=task)
        self.id2name = {0: "roi"}


class IndicatorLightRecognition(BaseOnnxInfer):
    """指示灯分类/特征模型：输出 roi 的 embedding 向量。"""

    def __init__(self, model_path: str, img_size: Tuple[int, int] = (224, 224), providers=None):
        super().__init__(model_path, providers=providers)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, im: np.ndarray):
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # -> NCHW
        # 分类模型无缩放语义，构造占位 meta 满足无状态推理链路约定
        meta = PreprocMeta(r=1.0, dw=0.0, dh=0.0, src_shape=im.shape)
        return img, meta

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
