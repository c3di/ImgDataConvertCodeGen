import os

from unittest.mock import patch, MagicMock
import math
import pytest
from src.imgdataconvertcodegen.code_exec.time_cost import time_cost, time_cost_in_kg
from src.imgdataconvertcodegen.knowledge_graph_construction import KnowledgeGraph, encode_metadata
from .data_for_tests.nodes_edges import test_edges


@pytest.fixture
def build_kg():
    kg = KnowledgeGraph()
    kg.load_from_file(
        os.path.join(os.path.dirname(__file__), "data_for_tests/kg_5nodes_4edges.json")
    )
    return kg


@patch(
    "src.imgdataconvertcodegen.code_exec.test_image_util.random_test_image_and_expected"
)
@pytest.mark.parametrize("source_node,target_node,conversion", test_edges)
def test_successful_execution(
    mock_random_test_image, source_node, target_node, conversion
):
    mock_random_test_image.return_value = (MagicMock(), MagicMock())
    result = time_cost(source_node, target_node, conversion)
    assert math.isfinite(result), "Result should be a finite number"


@patch(
    "src.imgdataconvertcodegen.code_exec.test_image_util.random_test_image_and_expected"
)
def test_time_cost_image_generation_exception(mock_random_test_image):
    mock_random_test_image.side_effect = Exception("Failed to generate image")
    result = time_cost(
        "source_node", "target_node", ("", "def convert(var):\n  return var")
    )
    assert (
        result == math.inf
    ), "time_cost should return math.inf when image generation fails"


#@patch("src.imgdataconvertcodegen.code_exec.test_image_util.random_test_image_and_expected")
#def test_time_cost_runtime_exception(mock_random_test_image):
#    mock_random_test_image.return_value = (MagicMock(), MagicMock())

#    invalid_conversion = (
#        "",
#        'def convert(var):\n    raise Exception("Conversion failed")\nconvert(source_image)'
#    )

#    with pytest.raises(Exception) as exc_info:
#        time_cost("source_node", "target_node", invalid_conversion)

#    assert "Conversion failed" in str(exc_info.value), "time_cost should raise RuntimeError with the correct message when conversion fails"


def test_time_cost_direct_call():
    with patch("src.imgdataconvertcodegen.code_exec.time_cost", return_value=1) as mock_time_cost:
        result = time_cost("source", "target", ("", "conversion code"))
        mock_time_cost.assert_called_once()
        assert result == 1


@patch("src.imgdataconvertcodegen.knowledge_graph_construction.encode_metadata")
@patch("src.imgdataconvertcodegen.code_exec.time_cost", return_value=1)
def test_time_cost_in_kg(mock_encode_metadata, mock_time_cost, build_kg):
    mock_encode_metadata.side_effect = encode_metadata
    kg = build_kg
    result = time_cost_in_kg(kg)

    expected_time_costs = {}
    for source, target in kg.edges:
        source_encoded = encode_metadata(source)
        target_encoded = encode_metadata(target)
        expected_time_costs[(source_encoded, target_encoded)] = 1

    #assert result == expected_time_costs, "Time costs calculated by time_cost_in_kg do not match expected values"
    assert mock_time_cost.call_count == len(kg.edges), "time_cost was not called the correct number of times."
