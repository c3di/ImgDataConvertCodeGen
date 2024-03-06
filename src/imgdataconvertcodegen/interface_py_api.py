from typing import Callable

from .code_generation import ConvertCodeGenerator
from .knowledge_graph_construction import get_knowledge_graph_builder

_builder = get_knowledge_graph_builder()
_code_generator = ConvertCodeGenerator(_builder.knowledge_graph)


def get_conversion(source_var_name: str, source_metadata: dict,
                   target_var_name: str, target_metadata: dict) -> str | None:
    """
    Generates Python code as a string that performs data conversion from a source variable to a target variable
    Args:
        source_var_name: the name of the variable holding the source data.
        source_metadata: A dictionary containing metadata about the source data, such as color channels, etc.
        target_var_name:  the name of the variable that will store the result of the conversion.
        target_metadata: the same as source_metadata

    Returns: A string containing the Python code necessary to perform the conversion.

    """
    return _code_generator.get_conversion(source_var_name, source_metadata,
                                          target_var_name, target_metadata)


def get_convert_path(source_metadata: dict, target_metadata: dict):
    return _code_generator.get_convert_path(source_metadata, target_metadata)


def add_image_metadata(new_metadata):
    _builder.add_new_metadata(new_metadata)
    _code_generator.knowledge_graph = _builder.knowledge_graph


def add_convert_code_factory(new_factory: Callable | str):
    _builder.add_new_edge_factory(new_factory)
    _code_generator.knowledge_graph = _builder.knowledge_graph
