"""请求示例与插件实际解析、输入转换保持一致。"""

import asyncio
import copy

import numpy as np

from vie_plugin_indicator_light.plugin import indicator_router


def test_document_example_matches_request_parser():
    router = indicator_router
    payload = copy.deepcopy(router.request_document_example)
    request = router.request_schema(payload)
    assert isinstance(request, router.request_document_model)
    assert request.model_dump(by_alias=True) == router.request_document_model.model_validate(payload).model_dump(by_alias=True)


def test_document_example_builds_inputs():
    router = indicator_router
    request = router.request_schema(copy.deepcopy(router.request_document_example))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    inputs = asyncio.run(router.get_inputs(request, image))
    registration = inputs.extra["registration"]
    assert registration["version"] == request.modelParams.type
    assert registration["model_file"] == request.AICameraModel[0].ModelFile
    assert registration["registration_id"] == request.AICameraModel[0].Id
    assert registration["register_mode"] is False
    assert inputs.image is image
