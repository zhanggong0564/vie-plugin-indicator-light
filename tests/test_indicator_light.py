"""indicator_light 插件单元测试：注册向量解析、embedding 比对与错误分类。"""
import numpy as np
import pytest
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import call, patch, MagicMock

from schemas.inference_context import InferenceContext
from schemas.data_base import IndicatorLightEmbedding
from schemas.exceptions import InvalidImageError, ModelInferenceError

_DEFAULT = object()


@pytest.fixture(autouse=True)
def _isolate_backflow_data(monkeypatch, tmp_path):
    from vie_plugin_indicator_light import business_logic

    monkeypatch.setattr(business_logic.settings, "DATA_DIR", str(tmp_path))


def _mock_runners():
    return patch(
        "vie_plugin_indicator_light.business_logic.create_inference_runner",
        side_effect=[MagicMock(), MagicMock()],
    )


@pytest.fixture
def api():
    """绕过真实模型、权重哈希和 Chroma 初始化。"""
    with (
        _mock_runners(),
        patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"),
        patch("vie_plugin_indicator_light.business_logic.pipeline_fingerprint", return_value="pipeline-v1"),
        patch("vie_plugin_indicator_light.business_logic._create_chroma_store"),
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver") as resolver_cls,
    ):
        from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
        instance = IndicatorLightBusinessAPI(MagicMock())
        instance.registration_resolver = resolver_cls.return_value
        yield instance


def _embedding_result(
    embeddings=_DEFAULT,
    boxes=_DEFAULT,
    scores=_DEFAULT,
    image_shape=(10, 10),
):
    return IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0]] if embeddings is _DEFAULT else embeddings,
        boxes=[[1, 1, 2, 2]] if boxes is _DEFAULT else boxes,
        scores=[0.9] if scores is _DEFAULT else scores,
        image_shape=image_shape,
    )


def _resolved_registration(result=None, image=None):
    result = result or _embedding_result()
    return SimpleNamespace(
        generation=SimpleNamespace(to_inference_result=MagicMock(return_value=result)),
        image=image,
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


def test_preprocess_descriptor_resolves_registered_embeddings(api):
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        extra={
            "registration": {
                "registration_id": "model-1",
                "material_no": "A001",
                "product_name": "indicator",
                "version": 1,
                "model_file": "https://example.com/reference.jpg",
                "create_time": None,
                "update_time": "2026-07-01T08:39:56",
                "register_mode": False,
            }
        },
    )
    api.registration_resolver.resolve.return_value = _resolved_registration()

    api.preprocess_hook(ctx)

    descriptor = api.registration_resolver.resolve.call_args.args[0]
    assert descriptor.registration_id == "model-1"
    assert descriptor.version == 1
    assert ctx.extra["registered_result"].embeddings == [[1.0, 0.0]]


def test_cache_hit_detect_infers_only_current_image(api):
    from schemas.data_base import InputParamsBusiness

    current_image = np.zeros((10, 10, 3), np.uint8)
    api.registration_resolver.resolve.return_value = _resolved_registration()
    api.detector.infer.return_value = _embedding_result()
    request_extra = {"registration": _descriptor_dict()}
    params = InputParamsBusiness(
        image=current_image,
        extra=request_extra,
    )

    result = api.detect(params)

    assert result.status is True
    api.detector.infer.assert_called_once_with(current_image)
    assert params.extra is request_extra
    assert params.extra == {"registration": _descriptor_dict()}
    assert "registered_result" not in params.extra


def test_preprocess_legacy_registered_image_infers_once(api):
    registered = np.ones((10, 10, 3), np.uint8)
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        registered=registered,
    )
    expected = _embedding_result()
    api.detector.infer.return_value = expected

    api.preprocess_hook(ctx)

    api.detector.infer.assert_called_once_with(registered)
    assert ctx.extra["registered_result"] is expected


@pytest.mark.parametrize("registration", [None, {}, {"registration_id": "only-id"}])
def test_preprocess_malformed_descriptor_raises_model_error(api, registration):
    extra = {} if registration is None else {"registration": registration}
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        extra=extra,
    )
    if registration is None:
        api.preprocess_hook(ctx)
        assert "registered_result" not in ctx.extra
    else:
        with pytest.raises(ModelInferenceError, match="注册描述"):
            api.preprocess_hook(ctx)


def test_preprocess_propagates_vision_api_error(api):
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        extra={"registration": _descriptor_dict()},
    )
    api.registration_resolver.resolve.side_effect = InvalidImageError("download failed")

    with pytest.raises(InvalidImageError, match="download failed"):
        api.preprocess_hook(ctx)


def test_preprocess_classifies_unexpected_resolver_error(api):
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        extra={"registration": _descriptor_dict()},
    )
    api.registration_resolver.resolve.side_effect = ValueError("bad embeddings")

    with pytest.raises(ModelInferenceError, match="注册向量解析失败"):
        api.preprocess_hook(ctx)


def test_business_post_process_all_match_without_registration_infer(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           registered=np.ones((10, 10, 3), np.uint8))
    ctx.raw_result = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
        image_shape=(10, 10),
    )
    ctx.extra["registered_result"] = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
        image_shape=(10, 10),
    )
    api.business_post_process(ctx)
    assert ctx.result.status is True
    assert len(ctx.result.detailList) == 2
    assert all(it.scene == "roi" for it in ctx.result.detailList)
    api.detector.infer.assert_not_called()


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
    ctx.extra["registered_result"] = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]], boxes=[[1, 1, 2, 2], [3, 3, 4, 4]], scores=[0.9, 0.8])
    api.business_post_process(ctx)
    assert ctx.result.status is False
    assert "Unable to match" in ctx.result.error_msg
    assert ctx.result.to_dict()["backflow_category"] == "unmatch"


def test_business_post_process_ignores_unmatched_extra_detection(api):
    ctx = InferenceContext(
        image=np.zeros((200, 300, 3), np.uint8),
        h=200,
        w=300,
        registered=np.ones((200, 300, 3), np.uint8),
    )
    ctx.raw_result = _embedding_result(
        embeddings=[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        boxes=[
            [90, 130, 110, 150],
            [250, 180, 270, 198],
            [30, 30, 50, 50],
            [150, 35, 170, 55],
        ],
        scores=[0.9, 0.8, 0.95, 0.92],
        image_shape=(200, 300),
    )
    ctx.extra["registered_result"] = _embedding_result(
        embeddings=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        boxes=[[20, 20, 40, 40], [120, 25, 140, 45], [70, 120, 90, 140]],
        scores=[0.9, 0.9, 0.9],
        image_shape=(200, 300),
    )

    api.business_post_process(ctx)

    assert ctx.result.status is True
    assert len(ctx.result.detailList) == 3
    assert [item.coordinate for item in ctx.result.detailList] == [
        [30, 30, 50, 50],
        [150, 35, 170, 55],
        [90, 130, 110, 150],
    ]


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
    ctx.extra["registered_result"] = IndicatorLightEmbedding(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
    )

    with pytest.raises(ModelInferenceError, match="boxes"):
        api.business_post_process(ctx)


def _descriptor_dict():
    return {
        "registration_id": "model-1",
        "material_no": "A001",
        "product_name": "indicator",
        "version": 1,
        "model_file": "https://example.com/reference.jpg",
        "create_time": None,
        "update_time": None,
        "register_mode": False,
    }


def test_initialization_uses_model_fingerprint_and_secure_download_options():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    with (
        _mock_runners(),
        patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"),
        patch("vie_plugin_indicator_light.business_logic.pipeline_fingerprint", return_value="fp") as fingerprint,
        patch("vie_plugin_indicator_light.business_logic._create_chroma_store") as store_factory,
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver") as resolver_cls,
    ):
        IndicatorLightBusinessAPI(MagicMock())

    fingerprint.assert_called_once_with(
        "./weights/indicator_light/rfdetr-small.onnx",
        "./weights/indicator_light/rec_v3.onnx",
    )
    store_factory.assert_called_once_with(
        "./data/indicator_light/chroma",
        "registered_embeddings",
        "fp",
    )
    kwargs = resolver_cls.call_args.kwargs
    assert kwargs["pipeline_fingerprint"] == "fp"
    assert kwargs["download_options"] == {
        "max_bytes": 20 * 1024 * 1024,
        "timeout": (3.0, 10.0),
        "allowed_hosts": (),
    }


def test_cache_disabled_skips_fingerprint_and_uses_null_store():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
    from vie_plugin_indicator_light.registration.store import NullRegistrationStore

    with (
        _mock_runners(),
        patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"),
        patch(
            "vie_plugin_indicator_light.business_logic.pipeline_fingerprint",
            side_effect=FileNotFoundError("weights are absent"),
        ) as fingerprint,
        patch.dict(
            os.environ,
            {"INDICATOR_VECTOR_CACHE_ENABLED": "false"},
        ),
        patch("vie_plugin_indicator_light.business_logic._create_chroma_store") as store_factory,
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver") as resolver_cls,
    ):
        IndicatorLightBusinessAPI(MagicMock())

    store_factory.assert_not_called()
    fingerprint.assert_not_called()
    assert isinstance(resolver_cls.call_args.kwargs["store"], NullRegistrationStore)
    assert resolver_cls.call_args.kwargs["pipeline_fingerprint"] == "cache-disabled"


def test_initialization_preserves_rec_v3_model_error_guidance():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    model_error = ModelInferenceError(
        "指示灯识别模型固定 batch 不支持，请使用或转换为 rec_v3.onnx"
    )
    with (
        _mock_runners(),
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec",
            side_effect=model_error,
        ),
    ):
        with pytest.raises(ModelInferenceError) as exc_info:
            IndicatorLightBusinessAPI(MagicMock())

    assert "rec_v3.onnx" in exc_info.value.error_msg
    assert exc_info.value.context["original_error"] is model_error


def test_cache_disabled_miss_detect_infers_registration_then_current():
    from schemas.data_base import InputParamsBusiness
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    current_image = np.zeros((10, 10, 3), np.uint8)
    registered_image = np.ones((10, 10, 3), np.uint8)
    detector = MagicMock()
    detector.infer.side_effect = [
        _embedding_result(),
        _embedding_result(),
    ]
    with (
        _mock_runners(),
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec",
            return_value=detector,
        ),
        patch(
            "vie_plugin_indicator_light.business_logic.pipeline_fingerprint",
            side_effect=FileNotFoundError("weights are absent"),
        ) as fingerprint,
        patch.dict(
            os.environ,
            {"INDICATOR_VECTOR_CACHE_ENABLED": "false"},
        ),
        patch(
            "vie_plugin_indicator_light.business_logic.download_image",
            return_value=registered_image,
        ) as downloader,
    ):
        api_instance = IndicatorLightBusinessAPI(MagicMock())
        result = api_instance.detect(
            InputParamsBusiness(
                image=current_image,
                extra={"registration": _descriptor_dict()},
            )
        )

    assert result.status is True
    fingerprint.assert_not_called()
    downloader.assert_called_once_with(
        _descriptor_dict()["model_file"],
        max_bytes=20 * 1024 * 1024,
        timeout=(3.0, 10.0),
        allowed_hosts=(),
    )
    assert detector.infer.call_args_list == [
        call(registered_image),
        call(current_image),
    ]


def test_legacy_detect_infers_registration_then_current():
    from schemas.data_base import InputParamsBusiness
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    current_image = np.zeros((10, 10, 3), np.uint8)
    registered_image = np.ones((10, 10, 3), np.uint8)
    detector = MagicMock()
    detector.infer.side_effect = [
        _embedding_result(),
        _embedding_result(),
    ]
    with (
        _mock_runners(),
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec",
            return_value=detector,
        ),
        patch.dict(
            os.environ,
            {"INDICATOR_VECTOR_CACHE_ENABLED": "false"},
        ),
    ):
        api_instance = IndicatorLightBusinessAPI(MagicMock())
        result = api_instance.detect(
            InputParamsBusiness(
                image=current_image,
                registered=registered_image,
            )
        )

    assert result.status is True
    assert detector.infer.call_args_list == [
        call(registered_image),
        call(current_image),
    ]


def test_chroma_init_failure_falls_back_to_null_store():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
    from vie_plugin_indicator_light.registration.store import NullRegistrationStore

    with (
        _mock_runners(),
        patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"),
        patch("vie_plugin_indicator_light.business_logic.pipeline_fingerprint", return_value="fp"),
        patch("vie_plugin_indicator_light.business_logic._create_chroma_store", side_effect=RuntimeError("secret")),
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver") as resolver_cls,
        patch("vie_plugin_indicator_light.business_logic.vision_logger.warning") as warning,
    ):
        IndicatorLightBusinessAPI(MagicMock())

    assert isinstance(resolver_cls.call_args.kwargs["store"], NullRegistrationStore)
    assert "RuntimeError" in str(warning.call_args)
    assert "secret" not in str(warning.call_args)


def test_chromadb_import_failure_falls_back_to_null_store():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
    from vie_plugin_indicator_light.registration.store import NullRegistrationStore

    with (
        _mock_runners(),
        patch("vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"),
        patch("vie_plugin_indicator_light.business_logic.pipeline_fingerprint", return_value="fp"),
        patch(
            "vie_plugin_indicator_light.business_logic._create_chroma_store",
            side_effect=ModuleNotFoundError("chromadb"),
        ),
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver") as resolver_cls,
    ):
        IndicatorLightBusinessAPI(MagicMock())

    assert isinstance(resolver_cls.call_args.kwargs["store"], NullRegistrationStore)


def test_business_logic_import_and_cache_disabled_init_do_not_require_chromadb():
    script = """
import importlib.abc
import os
import sys
from unittest.mock import MagicMock, patch

class BlockChroma(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "chromadb" or fullname.startswith("chromadb."):
            raise ModuleNotFoundError("blocked chromadb import")
        return None

sys.meta_path.insert(0, BlockChroma())
import vie_plugin_indicator_light.business_logic as business_logic
assert "chromadb" not in sys.modules
with (
    patch.object(
        business_logic,
        "create_inference_runner",
        side_effect=[MagicMock(), MagicMock()],
    ),
    patch.object(business_logic, "IndicatorLightDetRec"),
    patch.object(
        business_logic,
        "pipeline_fingerprint",
        side_effect=FileNotFoundError("weights are absent"),
    ) as fingerprint,
    patch.dict(
        os.environ,
        {"INDICATOR_VECTOR_CACHE_ENABLED": "false"},
    ),
):
    business_logic.IndicatorLightBusinessAPI(MagicMock())
fingerprint.assert_not_called()
assert "chromadb" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "../..:."
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(__file__) + "/..",
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


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
