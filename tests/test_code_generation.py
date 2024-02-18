from unittest.mock import patch, MagicMock

import pytest

from src.imgdataconvertcodegen.convert_code_generation import ConvertCodeGenerator
from src.imgdataconvertcodegen.knowledge_graph_construction import KnowledgeGraph
from test_data.test_nodes_edges_presets_for_kg import test_lib_preset


@pytest.fixture
def code_generator():
    kg = KnowledgeGraph(test_lib_preset)
    kg.load_from_file('test_data/test_kg_5nodes_4edges.json')
    return ConvertCodeGenerator(kg)


def test_convert_code_generator_init(code_generator):
    assert len(code_generator.knowledge_graph.nodes) == 5, \
        "Expected 5, but got " + str(code_generator.knowledge_graph.nodes)


def test_conversion_path(code_generator):
    kg = code_generator.knowledge_graph
    path = code_generator.get_convert_path(kg.get_node(1), kg.get_node(5))
    assert path == [kg.get_node(1), kg.get_node(3), kg.get_node(4), kg.get_node(5)]


def test_conversion_path_no_path(code_generator):
    kg = code_generator.knowledge_graph
    path = code_generator.get_convert_path(kg.get_node(5), kg.get_node(1))
    assert path == [], "Expected empty list, but got " + str(path)


def test_conversion_path_lib_preset(code_generator):
    kg = code_generator.knowledge_graph
    path = code_generator.get_convert_path('numpy', 'torch')
    assert path == [kg.get_node(1), kg.get_node(3), kg.get_node(4), kg.get_node(5)]


def test_generate_conversion_no_path(code_generator):
    kg = code_generator.knowledge_graph
    source_var = 'source_var'
    target_var = 'result'
    generated_code = code_generator.get_conversion(source_var, kg.get_node(5), target_var, kg.get_node(1))
    assert generated_code is None, "Expected None, but got " + str(generated_code)
    assert list(code_generator._cache.values()) == [None]


def test_generate_conversion_same_type(code_generator):
    kg = code_generator.knowledge_graph
    source_var = 'source_var'
    target_var = 'result'
    generated_code = code_generator.get_conversion(source_var, kg.get_node(1), target_var, kg.get_node(1))
    expected_code = f'{target_var} = {source_var}'

    assert generated_code == expected_code, "Expected " + expected_code + ", but got " + str(generated_code)
    assert list(code_generator._cache.values()) == [expected_code]


def test_generate_conversion_multiple_steps(code_generator):
    kg = code_generator.knowledge_graph
    source_var = 'source_var'
    target_var = 'result'
    with (patch('src.imgdataconvertcodegen.convert_code_generation.uuid.uuid4') as mock_uuid):
        mock_uuid.side_effect = [MagicMock(hex='first_uuid_hex'), MagicMock(hex='second_uuid_hex')]
        generated_code = code_generator.get_conversion(source_var, kg.get_node(1), target_var, kg.get_node(5))

        expected_code = ('import torch\n'
                         'var_first_uuid_hex = torch.from_numpy(source_var)\n'
                         'var_second_uuid_hex = var_first_uuid_hex.permute(2, 0, 1)\n'
                         'result = torch.unsqueeze(var_second_uuid_hex, 0)')

        assert generated_code == expected_code, f'Expected {expected_code}, but got {str(generated_code)}'
