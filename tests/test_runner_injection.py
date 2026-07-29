import os
from unittest.mock import MagicMock, call, patch

import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from services.scenario_registry import scenario_registry


def _cache_disabled():
    return patch.dict(
        os.environ,
        {"INDICATOR_VECTOR_CACHE_ENABLED": "false"},
    )


def test_scene_registers_with_scenario_registry():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    assert scenario_registry.snapshot()["indicator_light"] is IndicatorLightBusinessAPI


def test_business_initialization_creates_and_injects_both_runners():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    settings = MagicMock()
    detection_runner = MagicMock()
    recognition_runner = MagicMock()
    with (
        patch(
            "vie_plugin_indicator_light.business_logic.create_inference_runner",
            side_effect=[detection_runner, recognition_runner],
        ) as runner_factory,
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec"
        ) as pipeline_class,
        patch("vie_plugin_indicator_light.business_logic.RegistrationResolver"),
        _cache_disabled(),
    ):
        api = IndicatorLightBusinessAPI(settings)

    options = OnnxRuntimeOptions.from_settings(settings)
    assert runner_factory.call_args_list == [
        call(
            RunnerSpec(
                scenario="indicator_light",
                onnx_path="./weights/indicator_light/det_yolo_v2.onnx",
            ),
            options,
        ),
        call(
            RunnerSpec(
                scenario="indicator_light",
                onnx_path="./weights/indicator_light/rec_v3.onnx",
            ),
            options,
        ),
    ]
    pipeline_class.assert_called_once_with(
        detection_runner=detection_runner,
        recognition_runner=recognition_runner,
        confThreshold=0.25,
    )
    assert api.detector is pipeline_class.return_value


def test_second_runner_failure_closes_first_runner():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    detection_runner = MagicMock()
    with (
        patch(
            "vie_plugin_indicator_light.business_logic.create_inference_runner",
            side_effect=[detection_runner, RuntimeError("runner failed")],
        ),
        pytest.raises(Exception, match="indicator_light 模型加载失败"),
    ):
        IndicatorLightBusinessAPI(MagicMock())

    detection_runner.close.assert_called_once_with()


def test_pipeline_failure_closes_both_runners():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    runners = [MagicMock(), MagicMock()]
    with (
        patch(
            "vie_plugin_indicator_light.business_logic.create_inference_runner",
            side_effect=runners,
        ),
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec",
            side_effect=RuntimeError("pipeline failed"),
        ),
        pytest.raises(Exception, match="indicator_light 模型加载失败"),
    ):
        IndicatorLightBusinessAPI(MagicMock())

    for runner in runners:
        runner.close.assert_called_once_with()


def test_registration_failure_closes_pipeline_resources():
    from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI

    runners = [MagicMock(), MagicMock()]
    pipeline = MagicMock()
    with (
        patch(
            "vie_plugin_indicator_light.business_logic.create_inference_runner",
            side_effect=runners,
        ),
        patch(
            "vie_plugin_indicator_light.business_logic.IndicatorLightDetRec",
            return_value=pipeline,
        ),
        patch(
            "vie_plugin_indicator_light.business_logic.RegistrationResolver",
            side_effect=RuntimeError("resolver failed"),
        ),
        _cache_disabled(),
        pytest.raises(Exception, match="indicator_light 模型加载失败"),
    ):
        IndicatorLightBusinessAPI(MagicMock())

    pipeline.close.assert_called_once_with()
