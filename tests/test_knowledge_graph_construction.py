import os
import pytest
import networkx as nx
from unittest.mock import patch

from src.imgdataconvertcodegen.knowledge_graph_construction.knowledge_graph import KnowledgeGraph
from test_data.nodes_examples import node_examples
from test_data.edge_factory_examples import edge_factory_examples, new_edge_factory


@pytest.fixture
def knowledge_graph():
    return KnowledgeGraph(node_examples, edge_factory_examples)


def test_knowledge_graph_init(knowledge_graph):
    assert isinstance(knowledge_graph._graph, nx.DiGraph), "The graph object is not an instance of nx.DiGraph"
    assert len(knowledge_graph._graph.nodes) == len(node_examples), \
        f"Graph has {len(knowledge_graph._graph.nodes)} nodes, expected {len(node_examples)}"
    assert len(knowledge_graph._graph.edges) == 4, \
        f"Graph has {len(knowledge_graph._graph.edges)} edges, expected {4}"


def test_get_node(knowledge_graph):
    node = knowledge_graph.get_node(1)
    assert node == node_examples[0], f"Expected {node_examples[0]}, got {node}"


def test_add_node(knowledge_graph):
    new_node = {'name': 'Node3'}
    knowledge_graph.add_node(new_node)
    assert new_node == knowledge_graph.get_node(6), f"Expected {new_node}, got {knowledge_graph.get_node(6)}"


def test_add_edge(knowledge_graph):
    knowledge_graph.add_edge(3, 2, 'convert')
    assert len(knowledge_graph._graph.edges) == 5, f"Graph has {len(knowledge_graph._graph.edges)} edges, expected {5}"
    assert knowledge_graph.get_edge(3, 2)['conversion'] == 'convert'


def test_add_nodes(knowledge_graph):
    new_nodes = [{'name': 'Node3'}, {'name': 'Node4'}]
    knowledge_graph.add_nodes(new_nodes)
    assert len(knowledge_graph._graph.nodes) == 7, f"Graph has {len(knowledge_graph._graph.nodes)} nodes, expected {7}"


def test_add_new_edge_factory(knowledge_graph):
    knowledge_graph.add_new_edge_factory(new_edge_factory)
    assert len(knowledge_graph._edge_factories) == len(edge_factory_examples)
    assert len(knowledge_graph._graph.edges) == 5, f"Graph has {len(knowledge_graph._graph.edges)} edges, expected {5}"


def test_save_to_file(knowledge_graph):
    with patch('src.imgdataconvertcodegen.knowledge_graph_construction.knowledge_graph.save_graph') as mock_save_graph:
        expected_file_path = os.path.join('test_data', 'knowledge_graph_example.json')
        knowledge_graph.save_to_file(expected_file_path)
        mock_save_graph.assert_called_once_with(knowledge_graph._graph, expected_file_path)


def test_get_shortest_path(knowledge_graph):
    knowledge_graph.add_edge(2, 3, 'convert')
    path = knowledge_graph.get_shortest_path(knowledge_graph.get_node(1), knowledge_graph.get_node(5))
    assert path == [1, 3, 4, 5], f"Expected [1, 3, 4, 5], got {path}"


def test_get_shortest_path_no_path(knowledge_graph):
    path = knowledge_graph.get_shortest_path(knowledge_graph.get_node(5), knowledge_graph.get_node(1))
    assert path is None, f"Expected None, got {path}"


def test_get_shortest_path_same_node(knowledge_graph):
    path = knowledge_graph.get_shortest_path(knowledge_graph.get_node(1), knowledge_graph.get_node(1))
    assert path == [1], f"Expected [1], got {path}"
