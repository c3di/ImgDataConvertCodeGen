import os
from unittest.mock import patch

import networkx as nx
import pytest

from imgdataconvertcodegen.knowledge_graph_construction import lib_presets, KnowledgeGraph
from test_data.test_nodes_edges_presets_for_kg import test_nodes, test_edges, new_node, new_edge


@pytest.fixture
def kg():
    kg = KnowledgeGraph(lib_presets)
    for node in test_nodes:
        kg.add_node(node)
    for edge in test_edges:
        kg.add_edge(edge[0], edge[1], edge[2])
    return kg


def test_knowledge_graph_init(kg):
    assert isinstance(kg._graph, nx.DiGraph), "The graph object is not an instance of nx.DiGraph"
    assert kg._lib_presets == lib_presets, f"Expected {lib_presets}, got {kg._lib_presets}"


def test_add_new_node(kg):
    assert len(test_nodes) + 1 == kg.add_node(new_node), f"New node was not added to the graph"


def test_add_old_node(kg):
    node = test_nodes[0]
    node_id_in_kg = 1
    assert node_id_in_kg == kg.add_node(node), f"Old node was not added to the graph"


def test_is_node_exist(kg):
    assert kg.is_node_exist(test_nodes[0]), f"Expected True, got False"
    assert not kg.is_node_exist(new_node), f"Expected False, got True"


def get_node_id(kg):
    node_id = kg.get_node_id(test_nodes[0])
    assert node_id == 1, f"Expected 1, got {node_id}"

    assert kg.get_node_id(new_node) is None, f"Expected None, got {kg.get_node_id(new_node)}"


def test_get_node(kg):
    node = kg.get_node(1)
    assert node == test_nodes[0], f"Expected {test_nodes[0]}, got {node}"


def test_add_edge(kg):
    kg.add_node(new_node)
    kg.add_edge(new_edge[0], new_edge[1], new_edge[2])
    assert kg.get_edge_data(new_edge[0], new_edge[1])['conversion'] == new_edge[2], \
        f"Expected {new_edge[2]}, got {kg.get_edge_data(new_edge[0], new_edge[1])['conversion']}"


def test_get_edge(kg):
    expected_edge = test_edges[0]
    assert kg.get_edge_data(1, 2)['conversion'] == expected_edge[2], f"Expected {expected_edge[2]}, got {kg.get_edge_data(1, 2)}"


def test_edge_failure(kg):
    assert kg.get_edge_data(2, 3) is None, f"Expected None, got {kg.get_edge_data(1, 3)}"


def test_save_to_file(kg):
    with patch('imgdataconvertcodegen.knowledge_graph_construction.knowledge_graph.save_graph') as mock_save_graph:
        expected_file_path = os.path.join('test_data', 'knowledge_graph_example.json')
        kg.save_to_file(expected_file_path)
        mock_save_graph.assert_called_once_with(kg._graph, expected_file_path)


def test_update_uuid_after_load_from_file(kg):
    file_path = os.path.join(os.path.dirname(__file__), 'test_data/test_kg_5nodes_4edges.json')
    kg.load_from_file(file_path)
    assert kg._uuid == 5, f"Expected 5, got {kg._uuid}"


def test_add_lib_preset(kg):
    lib_name = 'new_lib'
    color_channel = 'color'
    metadata = new_node
    kg.add_lib_preset(lib_name, color_channel, metadata)
    assert kg._lib_presets[color_channel][
               lib_name] == metadata, f"Expected {metadata}, got {kg._lib_presets[color_channel][lib_name]}"


def test_add_lib_preset_failure(kg):
    lib_name = 'numpy'
    color_channel = 'color'
    metadata = new_node
    with pytest.raises(ValueError) as e:
        kg.add_lib_preset(lib_name, color_channel, metadata)
    assert str(
        e.value) == f"{lib_name} already in the lib_presets. We support {list(kg._lib_presets[color_channel].keys())}"


def test_get_metadata_by_lib_name(kg):
    lib_name = 'numpy'
    actual = kg.get_metadata_by_lib_name(lib_name)
    assert actual == lib_presets['color'][lib_name], f"Expected {lib_presets['color'][lib_name]}, got {actual}"


def test_get_metadata_by_lib_name_failure(kg):
    with pytest.raises(ValueError) as e:
        kg.get_metadata_by_lib_name('not_a_lib')
    assert str(e.value) == f"not_a_lib not in the lib_presets. We support {list(lib_presets['color'].keys())}"


def test_get_shortest_path(kg):
    kg.add_node(new_node)
    kg.add_edge(new_edge[0], new_edge[1], new_edge[2])
    path = kg.get_shortest_path(kg.get_node(1), kg.get_node(5))
    assert path == [1, 3, 4, 5], f"Expected [1, 3, 4, 5], got {path}"


def test_get_shortest_path_no_path(kg):
    path = kg.get_shortest_path(kg.get_node(3), kg.get_node(1))
    assert path is None, f"Expected None, got {path}"


def test_get_shortest_path_same_node(kg):
    path = kg.get_shortest_path(kg.get_node(1), kg.get_node(1))
    assert path == [1], f"Expected [1], got {path}"


def test_knowledge_graph_str(kg):
    expected_str = "Knowledge Graph with 4 nodes and 3 edges."
    assert str(kg) == expected_str, f"Expected {expected_str}, got {str(kg)}"