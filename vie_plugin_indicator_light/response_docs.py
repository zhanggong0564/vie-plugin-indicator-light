"""场景响应文档：合成示例，保留实际业务字段和判定语义。"""

from schemas.data_base import DetectionItem, MoMResult


RESPONSE_NOTES = (
    '### 场景明细与判定规则\n\n匹配成功后，每个明细代表当前图片中一个已匹配的指示灯位置，scene=roi，name 当前为空字符串；accuracy '
    '是映射到 [0,1] 的余弦相似度（保留三位小数），不是目标检测置信度或亮灯概率。单项按未舍入相似度严格大于配置阈值判为 PASS，全部匹配项通过才整体'
    '通过；默认阈值为 0.80，可由配置修改。数量或布局无法可靠匹配时返回 code=1、verdict=FAIL，并包含空坐标占位明细及 error_ms'
    'g 原因；这属于业务不匹配，不是执行异常，也不是 REVIEW。注册图获取或模型推理异常则按公共错误码返回 verdict=null。\n\n本场景默认 c'
    'oordinate 为原图宽高归一化的四边形八个数：[x1,y1,x2,y2,x3,y3,x4,y4]，矩形按左上、右上、右下、左下排列；x 乘原图宽、'
    'y 乘原图高可还原像素。缺失项可以为 []，不得当作原点检测框。color 是显示颜色，不作为判定依据；以 verdict 为准。vis_image 为'
    '可选 JPEG data URI，关闭可视化或绘制失败时可为空；示例使用空字符串，不填伪造 base64。 当前这些示例场景的公共 ocr_tokens'
    ' 为 null；不要把内部 OCR 结构当作既有接口字段。'
)


def _result(passed):
    items = [DetectionItem(
        status=passed or index == 0, scene="roi", accuracy=0.98 if passed or index == 0 else 0.4,
        coordinate=[0.1 + index * 0.3, 0.2, 0.2 + index * 0.3, 0.2,
                    0.2 + index * 0.3, 0.3, 0.1 + index * 0.3, 0.3],
    ) for index in range(2)]
    for item in items:
        item.coordinate = [round(value, 4) for value in item.coordinate]
    return MoMResult(status=passed, message="success" if passed else "failed", detailList=items).to_dict()


RESPONSE_EXAMPLES = {
    "PASS": {"summary": "通过：两个位置均与注册参考图相似", "result": _result(True)},
    "FAIL": {"summary": "不通过：一个位置相似度低于阈值", "result": _result(False)},
    "UNMATCH": {"summary": "业务不匹配：当前图与注册图数量不一致", "result": MoMResult(
        status=False, message="失败",
        error_msg="Unable to match all registered indicator positions 2!=1.",
        detailList=[DetectionItem(status=False)],
    ).to_dict()},
}
