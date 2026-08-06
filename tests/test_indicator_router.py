import asyncio
import importlib
import importlib.util
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import requests

from schemas.exceptions import InvalidParamsError

_MISSING = object()


def _restore_entry(mapping, key, previous):
    if previous is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = previous


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("router must not perform network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "getaddrinfo", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)

    try:
        import httpx
    except ImportError:
        pass
    else:
        monkeypatch.setattr(httpx.Client, "request", fail_network)
        monkeypatch.setattr(httpx.AsyncClient, "request", fail_network)


@pytest.fixture
def plugin_module(monkeypatch, _forbid_network):
    package = importlib.import_module("vie_plugin_indicator_light")
    plugin_name = "vie_plugin_indicator_light.plugin"
    business_logic_name = "vie_plugin_indicator_light.business_logic"
    package_vars = vars(package)
    previous = {
        "plugin_module": sys.modules.get(plugin_name, _MISSING),
        "plugin_attr": package_vars.get("plugin", _MISSING),
        "business_logic_module": sys.modules.get(business_logic_name, _MISSING),
        "business_logic_attr": package_vars.get("business_logic", _MISSING),
    }

    sys.modules.pop(plugin_name, None)
    package_vars.pop("plugin", None)
    package_vars.pop("business_logic", None)

    try:
        with monkeypatch.context() as modules:
            business_logic_stub = types.ModuleType(business_logic_name)
            modules.setitem(sys.modules, business_logic_name, business_logic_stub)

            if importlib.util.find_spec("python_multipart") is None:
                multipart_stub = types.ModuleType("python_multipart")
                multipart_stub.__version__ = "0.0.20"
                modules.setitem(sys.modules, "python_multipart", multipart_stub)

            yield importlib.import_module(plugin_name)
    finally:
        _restore_entry(sys.modules, plugin_name, previous["plugin_module"])
        _restore_entry(package_vars, "plugin", previous["plugin_attr"])
        _restore_entry(
            sys.modules,
            business_logic_name,
            previous["business_logic_module"],
        )
        _restore_entry(
            package_vars,
            "business_logic",
            previous["business_logic_attr"],
        )

        assert sys.modules.get(plugin_name, _MISSING) is previous["plugin_module"]
        assert package_vars.get("plugin", _MISSING) is previous["plugin_attr"]
        assert (
            sys.modules.get(business_logic_name, _MISSING)
            is previous["business_logic_module"]
        )
        assert (
            package_vars.get("business_logic", _MISSING)
            is previous["business_logic_attr"]
        )


def _model(
    *,
    model_id: str = "df43d912b4794f13b11d7bc08720b089",
    version: int = 1,
    model_file: str = "http://10.172.2.32:9986/ProblemRecord/reference.jpg",
    product_name: str = "A0SW1821",
    create_time: str | None = "2026-07-01T08:39:56",
    update_time: str | None = "2026-07-01T08:39:56",
) -> dict:
    return {
        "Id": model_id,
        "SN": "A2662503392",
        "ProductName": product_name,
        "Version": version,
        "AIProductTypeName": "风电检验组",
        "AIProductTypeValue": "风电检验组",
        "ModelFile": model_file,
        "Remark": None,
        "CreateBy": None,
        "CreateTime": create_time,
        "UpdateBy": None,
        "UpdateTime": update_time,
        "AIParameterName": "产品类型",
        "AIParameterValue": "五路无熔丝盒无磁环",
        "DictionaryCode": None,
    }


_UNSET = object()


def _request(
    plugin_module,
    models: list[dict] | None | object = _UNSET,
    *,
    version: int = 1,
    register: str | None | object = "false",
):
    model_params = {"type": str(version)}
    if register is not _UNSET:
        model_params["register"] = register
    payload = {
        "modelParams": model_params,
        "type": "A0SW1821",
        "product": "双馈风电变流器_WG10MDF-1140-AZ-N-V1612_10MW_A_无",
    }
    if models is not _UNSET:
        payload["AICameraModel"] = models
    return plugin_module.indicator_router.request_schema(payload)


@pytest.mark.parametrize(
    ("register", "expected"),
    [
        (_UNSET, None),
        (None, None),
        ("true", True),
        ("false", False),
    ],
)
def test_get_inputs_preserves_optional_register_mode_without_downloading(
    plugin_module, register, expected
):
    request = _request(plugin_module, [_model()], register=register)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    inputs = asyncio.run(plugin_module.indicator_router.get_inputs(request, image))

    assert not hasattr(plugin_module, "download_image")
    assert not hasattr(plugin_module, "run_sync")
    assert inputs.image is image
    assert inputs.product_type == "1"
    assert inputs.registered.size == 0
    assert inputs.extra == {
        "registration": {
            "registration_id": "df43d912b4794f13b11d7bc08720b089",
            "material_no": "A0SW1821",
            "product_name": "A0SW1821",
            "version": 1,
            "model_file": "http://10.172.2.32:9986/ProblemRecord/reference.jpg",
            "create_time": "2026-07-01T08:39:56",
            "update_time": "2026-07-01T08:39:56",
            "register_mode": expected,
        }
    }


@pytest.mark.parametrize(
    "models",
    [
        _UNSET,
        None,
        [],
        [_model(version=2)],
        [
            _model(model_id="valid"),
            _model(model_id="last", model_file=""),
        ],
    ],
    ids=["omitted", "null", "empty", "version-mismatch", "last-match-empty-file"],
)
def test_get_inputs_preserves_no_matching_model_error(plugin_module, models):
    request = _request(plugin_module, models, version=1)

    with pytest.raises(InvalidParamsError) as exc_info:
        asyncio.run(
            plugin_module.indicator_router.get_inputs(
                request,
                np.zeros((2, 2, 3), dtype=np.uint8),
            )
        )

    assert exc_info.value.error_msg == "未找到型号 1 对应的注册参考图"


def test_get_inputs_uses_last_matching_model(plugin_module):
    request = _request(
        plugin_module,
        [
            _model(
                model_id="first",
                model_file="http://example.com/first.jpg",
                product_name="first-product",
            ),
            _model(version=2, model_id="different-version"),
            _model(
                model_id="last",
                model_file="http://example.com/last.jpg",
                product_name="last-product",
                create_time="2026-07-02T01:02:03",
                update_time=None,
            ),
        ]
    )

    inputs = asyncio.run(
        plugin_module.indicator_router.get_inputs(
            request,
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )

    assert inputs.extra["registration"] == {
        "registration_id": "last",
        "material_no": "A0SW1821",
        "product_name": "last-product",
        "version": 1,
        "model_file": "http://example.com/last.jpg",
        "create_time": "2026-07-02T01:02:03",
        "update_time": None,
        "register_mode": False,
    }


def test_backflow_target_uses_filename_timestamp(plugin_module):
    target = plugin_module.indicator_router.resolve_backflow_target(
        "风电-整机组装1-231-1782460558709.jpg",
        "A0SW1821/1",
    )

    assert target.save_stem == "1782460558709"
    assert target.model_dir == "A0SW1821"
    assert target.model_subdir == "1"


@pytest.mark.parametrize(
    ("material_no", "version", "expected"),
    [
        ("A0SW2163", 1, "A0SW2163/1"),
        ("A0SW2163-1", 1, "A0SW2163/1"),
        ("A0SW2163-2", 2, "A0SW2163/2"),
    ],
)
def test_backflow_key_separates_material_and_version(
    plugin_module, material_no, version, expected
):
    request = _request(
        plugin_module,
        [_model(version=version)],
        version=version,
    )
    request.type = material_no

    assert plugin_module.indicator_router._extract_product_type(request) == expected


def test_backflow_paths_include_material_version_directory(plugin_module):
    paths = plugin_module.indicator_router.backflow_service.resolve_paths(
        "风电-1782460558709.jpg",
        "2026-08-06T10:00:00.000",
        "A0SW2163/2",
        "unmatch",
        ".jpg",
    )

    assert Path(paths["image_path"]).parts[-6:] == (
        "2026-08-06",
        "A0SW2163",
        "2",
        "unmatch",
        "images",
        "1782460558709.jpg",
    )


def test_backflow_count_mismatch_uses_unmatch_directory(plugin_module):
    assert plugin_module.indicator_router.backflow_service.classify_result(
        {"status": "false", "backflow_category": "unmatch"}
    ) == "unmatch"


def test_backflow_regular_failure_still_uses_ng_directory(plugin_module):
    assert plugin_module.indicator_router.backflow_service.classify_result(
        {"status": "false"}
    ) == "ng"
