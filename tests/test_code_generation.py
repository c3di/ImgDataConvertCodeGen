import os.path
from unittest.mock import patch, MagicMock

import pytest

from imgdataconvertcodegen.code_generation import ConvertCodeGenerator
from imgdataconvertcodegen.knowledge_graph_construction import KnowledgeGraph
from data_for_tests.nodes_edges_presets_for_kg import test_lib_preset


@pytest.fixture
def code_generator():
    kg = KnowledgeGraph(test_lib_preset)
    kg.load_from_file(os.path.join(os.path.dirname(__file__), 'data_for_tests/kg_5nodes_4edges.json'))
    return ConvertCodeGenerator(kg)


def test_convert_code_generator_init(code_generator):
    assert len(code_generator.knowledge_graph.nodes) == 5, \
        "Expected 5, but got " + str(code_generator.knowledge_graph.nodes)


def test_knowledge_graph_property(code_generator):
    new_kg = KnowledgeGraph(test_lib_preset)
    code_generator.knowledge_graph = new_kg
    assert code_generator.knowledge_graph == new_kg, (
            "Expected " + str(new_kg) + ", but got " + str(code_generator.knowledge_graph))


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
    with (patch('imgdataconvertcodegen.code_generation.uuid.uuid4') as mock_uuid):
        mock_uuid.side_effect = [MagicMock(hex='first_uuid_hex'), MagicMock(hex='second_uuid_hex')]
        generated_code = code_generator.get_conversion(source_var, kg.get_node(1), target_var, kg.get_node(5))

        expected_code = ('import torch\n'
                         'var_first_uuid_hex = torch.from_numpy(source_var)\n'
                         'var_second_uuid_hex = var_first_uuid_hex.permute(2, 0, 1)\n'
                         'result = torch.unsqueeze(var_second_uuid_hex, 0)')

        assert generated_code == expected_code, f'Expected {expected_code}, but got {str(generated_code)}'


def test_generate_conversion_using_cache(code_generator):
    kg = code_generator.knowledge_graph
    source_var = 'source_var'
    target_var = 'result'
    with (patch('imgdataconvertcodegen.code_generation.uuid.uuid4') as mock_uuid):
        mock_uuid.side_effect = [MagicMock(hex='first_uuid_hex'), MagicMock(hex='second_uuid_hex')]
        generated_code = code_generator.get_conversion(source_var, kg.get_node(1), target_var, kg.get_node(5))
        assert list(code_generator._cache.values()) == [generated_code], f"Code not cached"

        code_from_cache = code_generator.get_conversion(source_var, kg.get_node(1), target_var, kg.get_node(5))
        expected_code = ('import torch\n'
                         'var_first_uuid_hex = torch.from_numpy(source_var)\n'
                         'var_second_uuid_hex = var_first_uuid_hex.permute(2, 0, 1)\n'
                         'result = torch.unsqueeze(var_second_uuid_hex, 0)')

        assert code_from_cache == expected_code, f'Expected {expected_code}, but got {str(code_from_cache)}'
