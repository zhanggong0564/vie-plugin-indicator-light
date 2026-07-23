"""indicator_light 插件用法示例（双输入：当前图 + 注册参考图）。

运行（从仓库根目录）：
    python plugins/vie-plugin-indicator-light/examples/run.py <当前图> <注册参考图>

前置：已 `pip install -e plugins/vie-plugin-indicator-light`；
      权重 ./weights/indicator_light/det_yolo_v2.onnx、indicator_light/rec_v3.onnx 就位。
"""
import os
import sys
import json

import cv2
import numpy as np

# 让示例在任意 cwd 下都能 import 框架（services/schemas 在仓库根，未作为包安装）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import vie_plugin_indicator_light.plugin  # noqa: E402,F401  触发 ScenarioRegistry 注册
from services.scenario_registry import scenario_registry  # noqa: E402
from schemas.data_base import InputParamsBusiness  # noqa: E402


TITLE_HEIGHT = 60
REGISTERED_COLOR = (255, 128, 0)
CURRENT_COLOR = (0, 255, 0)
MISMATCH_COLOR = (0, 0, 255)


def _draw_detections(image, boxes, scores, color):
    canvas = image.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(value) for value in box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        score = scores[index] if index < len(scores) else None
        label = f"{index + 1}"
        if score is not None:
            label += f" {float(score):.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _resize_to_height(image, target_height):
    height, width = image.shape[:2]
    target_width = max(1, round(width * target_height / height))
    return cv2.resize(image, (target_width, target_height))


def _build_panel(image, boxes, scores, title, color, target_height):
    annotated = _draw_detections(image, boxes, scores, color)
    annotated = _resize_to_height(annotated, target_height)
    title_bar = np.full(
        (TITLE_HEIGHT, annotated.shape[1], 3),
        32,
        dtype=np.uint8,
    )
    cv2.putText(
        title_bar,
        f"{title} | ROI: {len(boxes)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack((title_bar, annotated))


def build_comparison_image(
    registered,
    registered_boxes,
    registered_scores,
    current,
    current_boxes,
    current_scores,
    max_height=1080,
):
    """绘制注册图和待检测图的原始 ROI，并按相同高度左右拼接。"""
    target_height = min(max_height, registered.shape[0], current.shape[0])
    registered_panel = _build_panel(
        registered,
        registered_boxes,
        registered_scores,
        "Registered",
        REGISTERED_COLOR,
        target_height,
    )
    current_panel = _build_panel(
        current,
        current_boxes,
        current_scores,
        "Current",
        CURRENT_COLOR,
        target_height,
    )
    comparison = np.hstack((registered_panel, current_panel))
    if len(registered_boxes) != len(current_boxes):
        cv2.putText(
            comparison,
            (
                f"ROI count mismatch: {len(registered_boxes)} "
                f"!= {len(current_boxes)}"
            ),
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            MISMATCH_COLOR,
            2,
            cv2.LINE_AA,
        )
    return comparison


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法: run.py <当前图> <注册参考图>")
    image = cv2.imread(sys.argv[1])
    registered = cv2.imread(sys.argv[2])
    if image is None or registered is None:
        raise SystemExit("无法读取图片（当前图或注册参考图）")
    detector = scenario_registry.create("indicator_light")
    try:
        result = detector.detect(InputParamsBusiness(image=image, registered=registered))
        out = result.to_dict()
        print(json.dumps(out, ensure_ascii=False, indent=2))

        # 独立获取两张图的原始 ROI，避免数量不一致时业务结果无坐标而无法诊断。
        registered_result = detector.detector.infer(registered)
        current_result = detector.detector.infer(image)
        comparison = build_comparison_image(
            registered,
            registered_result.boxes or [],
            registered_result.scores or [],
            image,
            current_result.boxes or [],
            current_result.scores or [],
        )
        save_path = "indicator_light_result.jpg"
        cv2.imwrite(save_path, comparison)
        print(f"可视化结果已保存: {save_path}")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
