"""indicator_light 插件单元测试：embedding 比对、双输入流程、缺注册图、数量不匹配与错误分类。"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from schemas.inference_context import InferenceContext
from schemas.data_base import IndicatorLightEmbedding
from schemas.exceptions import ModelInferenceError

_DEFAULT = object()


@pytest.fixture
def api():
    """绕过真实模型加载（detector 被 mock），sim_thr 由插件 config 提供(0.7)。"""
    with patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"):
        from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
        yield IndicatorLightBusinessAPI(MagicMock())


def _embedding_result(embeddings=_DEFAULT, boxes=_DEFAULT, scores=_DEFAULT):
    return IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0]] if embeddings is _DEFAULT else embeddings,
        boxes=[[1, 1, 2, 2]] if boxes is _DEFAULT else boxes,
        scores=[0.9] if scores is _DEFAULT else scores,
    )


def test_compare_embedding_identical(api):
    item = api.compare_embedding([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert item.status is True        # cosine=1 → 1.0 > 0.7
    assert item.accuracy == 1.0
    assert item.scene == "roi"


def test_compare_embedding_orthogonal(api):
    item = api.compare_embedding([1.0, 0.0], [0.0, 1.0])
    assert item.status is False       # cosine=0 → 0.5 < 0.7
    assert item.accuracy == 0.5


def test_business_post_process_all_match(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           registered=np.ones((10, 10, 3), np.uint8))
    ctx.raw_result = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
    )
    # 注册图推理返回与当前图完全一致的 embedding → 全部匹配
    api.detector.infer.return_value = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
    )
    api.business_post_process(ctx)
    assert ctx.result.status is True
    assert len(ctx.result.detailList) == 2
    assert all(it.scene == "roi" for it in ctx.result.detailList)


def test_business_post_process_missing_registered(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10, registered=None)
    ctx.raw_result = IndicatorLightEmbedding(embeddings=[[1.0]], boxes=[[1, 1, 2, 2]], scores=[0.9])
    api.business_post_process(ctx)
    assert ctx.result.status is False
    assert "registered" in ctx.result.error_msg


def test_business_post_process_count_mismatch(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           registered=np.ones((10, 10, 3), np.uint8))
    ctx.raw_result = IndicatorLightEmbedding(embeddings=[[1.0, 0.0]], boxes=[[1, 1, 2, 2]], scores=[0.9])
    api.detector.infer.return_value = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]], boxes=[[1, 1, 2, 2], [3, 3, 4, 4]], scores=[0.9, 0.8])
    api.business_post_process(ctx)
    assert ctx.result.status is False
    assert "does not match" in ctx.result.error_msg


def test_compare_embedding_zero_vector_raises_model_error(api):
    with pytest.raises(ModelInferenceError, match="空向量"):
        api.compare_embedding([0.0, 0.0], [1.0, 0.0])


def test_compare_embedding_shape_mismatch_raises_model_error(api):
    with pytest.raises(ModelInferenceError, match="维度"):
        api.compare_embedding([1.0, 0.0, 0.0], [1.0, 0.0])


def test_business_post_process_boxes_mismatch_raises_model_error(api):
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        registered=np.ones((10, 10, 3), np.uint8),
    )
    ctx.raw_result = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2]],
        scores=[0.9, 0.8],
    )
    api.detector.infer.return_value = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
    )

    with pytest.raises(ModelInferenceError, match="boxes"):
        api.business_post_process(ctx)


def test_compare_none_current_result_raises_model_error(api):
    registered = _embedding_result()

    with pytest.raises(ModelInferenceError, match="当前图"):
        api._compare(None, registered)


def test_compare_none_registered_result_raises_model_error(api):
    current = _embedding_result()

    with pytest.raises(ModelInferenceError, match="注册"):
        api._compare(current, None)


def test_compare_missing_current_embeddings_raises_model_error(api):
    current = _embedding_result(embeddings=None)
    registered = _embedding_result()

    with pytest.raises(ModelInferenceError, match="当前图特征"):
        api._compare(current, registered)


def test_compare_missing_registered_embeddings_raises_model_error(api):
    current = _embedding_result()
    registered = _embedding_result(embeddings=None)

    with pytest.raises(ModelInferenceError, match="注册标准特征"):
        api._compare(current, registered)


def test_compare_boxes_none_raises_model_error(api):
    current = _embedding_result(boxes=None)
    registered = _embedding_result()

    with pytest.raises(ModelInferenceError, match="boxes"):
        api._compare(current, registered)
