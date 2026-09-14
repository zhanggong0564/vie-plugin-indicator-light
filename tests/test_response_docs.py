"""文档相似度示例与真实比对逻辑保持一致。"""
import numpy as np
import pytest

from vie_plugin_indicator_light.business_logic import IndicatorLightBusinessAPI
from vie_plugin_indicator_light.response_docs import RESPONSE_EXAMPLES


@pytest.mark.parametrize("name", ["PASS", "FAIL"])
def test_documented_similarity_matches_comparison(name):
    api = object.__new__(IndicatorLightBusinessAPI)
    api.sim_thr = 0.65
    for item in RESPONSE_EXAMPLES[name]["result"]["detailList"]:
        cosine = item["accuracy"] * 2 - 1
        actual = api.compare_embedding([1.0, 0.0], [cosine, np.sqrt(1 - cosine ** 2)]).to_dict()
        assert actual["accuracy"] == item["accuracy"]
        assert actual["verdict"] == item["verdict"]
        assert actual["scene"] == item["scene"]


def test_documented_unmatch_is_business_failure():
    actual = IndicatorLightBusinessAPI._unmatch_result(2, 1).to_dict()
    expected = RESPONSE_EXAMPLES["UNMATCH"]["result"]
    for key in expected:
        assert actual[key] == expected[key]
