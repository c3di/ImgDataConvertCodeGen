import pytest
from unittest.mock import MagicMock, patch
from src.imgdataconvertcodegen.convert_code_generation import ConvertCodeGenerator


@pytest.fixture
def mock_knowledge_graph():
    mock_kg = MagicMock()
    mock_kg.get_shortest_path.return_value = [1, 2, 3]
    mock_kg.get_edge.side_effect = lambda x, y: {'conversion': f'convert_{x}_to_{y}'}
    return mock_kg


def test_conversion_functions_path_exists(mock_knowledge_graph):
    converter = ConvertCodeGenerator(mock_knowledge_graph)
    functions = converter.conversion_functions('source_metadata', 'target_metadata')
    assert functions == ['convert_1_to_2', 'convert_2_to_3']


def test_conversion_functions_no_path(mock_knowledge_graph):
    mock_knowledge_graph.get_shortest_path.return_value = None
    converter = ConvertCodeGenerator(mock_knowledge_graph)
    functions = converter.conversion_functions('source_metadata', 'target_metadata')
    assert functions is None


@patch('src.imgdataconvertcodegen.convert_code_generation.create_unique_function')
def test_generate_code(mock_create_unique_function, mock_knowledge_graph):
    mock_create_unique_function.side_effect = lambda func: {"function_name": f'func_{func}',
                                                            "function_definition": f'def {func}(): pass'}

    converter = ConvertCodeGenerator(mock_knowledge_graph)
    source_var = 'source_var'
    target_var = 'result'
    generated_code = converter.generate_code(source_var, 'source_metadata', target_var, 'target_metadata')

    expected_definitions = 'def convert_1_to_2(): pass\ndef convert_2_to_3(): pass'
    expected_code = f'{expected_definitions}\n{target_var} = func_convert_2_to_3(func_convert_1_to_2({source_var}))'

    assert generated_code == expected_code
