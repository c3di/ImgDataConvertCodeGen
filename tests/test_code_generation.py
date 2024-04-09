import os.path
from unittest.mock import MagicMock, patch

import pytest

from src.imgdataconvertcodegen.code_generator import ConvertCodeGenerator
from src.imgdataconvertcodegen.knowledge_graph_construction import KnowledgeGraph
from .data_for_tests.nodes_edges import new_node, test_nodes, all_nodes


@pytest.fixture
def code_generator():
    kg = KnowledgeGraph()
    kg.load_from_file(
        os.path.join(os.path.dirname(__file__), "data_for_tests/kg_5nodes_4edges.json")
    )
    return ConvertCodeGenerator(kg)


def test_convert_code_generator_init(code_generator):
    assert len(code_generator.knowledge_graph.nodes) == 5, "Expected 5, but got " + str(
        code_generator.knowledge_graph.nodes
    )


def test_knowledge_graph_property(code_generator):
    new_kg = KnowledgeGraph()
    code_generator.knowledge_graph = new_kg
    assert code_generator.knowledge_graph == new_kg, (
        "Expected " + str(new_kg) + ", but got " + str(code_generator.knowledge_graph)
    )


def test_conversion_path(code_generator):
    expected_path = ["node1", "node2"]
    code_generator.knowledge_graph.get_shortest_path = MagicMock()
    code_generator.knowledge_graph.get_shortest_path.return_value = expected_path
    actual = code_generator.get_convert_path(
        {"source": "source_metadata"}, {"target": "target_metadata"}
    )

    code_generator.knowledge_graph.get_shortest_path.assert_called_once_with(
        {"source": "source_metadata"},
        {"target": "target_metadata"},
        code_generator._goal_function_for_AStar,
    )
    assert (
        actual == expected_path
    ), "The returned path does not match the expected path."


def test_generate_conversion_no_path(code_generator):
    source_var = "source_var"
    target_var = "result"
    generated_code = code_generator.get_conversion(
        source_var, new_node, target_var, test_nodes[0]
    )
    assert generated_code is None, "Expected None, but got " + str(generated_code)
    assert list(code_generator._cache.values()) == [None]


def test_generate_conversion_same_type(code_generator):
    source_var = "source_var"
    target_var = "result"
    generated_code = code_generator.get_conversion(
        source_var, test_nodes[0], target_var, test_nodes[0]
    )
    expected_code = f"{target_var} = {source_var}"

    assert generated_code == expected_code, (
        "Expected " + expected_code + ", but got " + str(generated_code)
    )
    assert list(code_generator._cache.values()) == [[test_nodes[0]]]


def test_generate_conversion_multiple_steps(code_generator):
    source_var = "source_var"
    target_var = "result"
    generated_code = code_generator.get_conversion(
        source_var, test_nodes[0], target_var, new_node
    )

    expected_code = (
        "import torch\n"
        "image = torch.from_numpy(source_var)\n"
        "image = image.permute(2, 0, 1)\n"
        "result = torch.unsqueeze(image, 0)"
    )

    assert (
        generated_code == expected_code
    ), f"Expected {expected_code}, but got {str(generated_code)}"


def test_generate_conversion_using_cache(code_generator):
    source_var = "source_var"
    target_var = "result"

    code_generator.get_conversion(source_var, test_nodes[0], target_var, new_node)
    assert list(code_generator._cache.values()) == [
        [all_nodes[0], all_nodes[2], all_nodes[3], all_nodes[4]]
    ], "Code not cached"

    code_from_cache = code_generator.get_conversion(
        source_var, test_nodes[0], target_var, new_node
    )
    expected_code = (
        "import torch\n"
        "image = torch.from_numpy(source_var)\n"
        "image = image.permute(2, 0, 1)\n"
        "result = torch.unsqueeze(image, 0)"
    )

    assert (
        code_from_cache == expected_code
    ), f"Expected {expected_code}, but got {str(code_from_cache)}"


@patch("src.imgdataconvertcodegen.code_generator.time_cost_in_kg")
def test_config_astar_goal_include_time_cost_max_nonzero(
    mock_time_cost_in_kg, code_generator
):
    cpu_penalty = 1.0
    gpu_penalty = 2.0
    include_time_cost = True
    test_img_size = (256, 256)

    mock_time_cost_in_kg.return_value = {
        ("encoded_node1", "encoded_node2"): 0.5,
        ("encoded_node2", "encoded_node3"): 0.3,
    }
    max_time_cost = 0.5

    code_generator.config_astar_goal_function(
        cpu_penalty, gpu_penalty, include_time_cost
    )
    normalized_cost = code_generator._normalize_time_cost(
        "encoded_node1", "encoded_node2"
    )
    expected_normalized_cost = round(0.5 / max_time_cost, 3)

    assert (
        normalized_cost == expected_normalized_cost
    ), f"Expected normalized cost to be {expected_normalized_cost}, got {normalized_cost}"

    mock_time_cost_in_kg.assert_called_once_with(
        code_generator.knowledge_graph, test_img_size
    )


@patch("src.imgdataconvertcodegen.code_generator.time_cost_in_kg")
def test_config_astar_goal_include_time_cost_max_zero(
    mock_time_cost_in_kg, code_generator
):
    cpu_penalty = 1.0
    gpu_penalty = 2.0
    include_time_cost = True
    test_img_size = (256, 256)

    mock_time_cost_in_kg.return_value = {
        ("encoded_node1", "encoded_node2"): 0.0,
        ("encoded_node2", "encoded_node3"): 0.0,
    }

    code_generator.config_astar_goal_function(
        cpu_penalty, gpu_penalty, include_time_cost
    )
    normalized_cost = code_generator._normalize_time_cost(
        "encoded_node1", "encoded_node2"
    )

    assert (
        normalized_cost == 0
    ), f"Expected normalized cost to be 0, got {normalized_cost}"

    mock_time_cost_in_kg.assert_called_once_with(
        code_generator.knowledge_graph, test_img_size
    )


def test_config_astar_goal_function_no_time_cost(code_generator):
    cpu_penalty = 1.0
    gpu_penalty = 2.0
    include_time_cost = False

    code_generator.config_astar_goal_function(
        cpu_penalty, gpu_penalty, include_time_cost
    )

    arbitrary_source = "any_source_node"
    arbitrary_target = "any_target_node"
    normalized_cost = code_generator._normalize_time_cost(
        arbitrary_source, arbitrary_target
    )

    assert (
        normalized_cost == 0
    ), "Expected normalized cost to be 0 when include_time_cost is False"
