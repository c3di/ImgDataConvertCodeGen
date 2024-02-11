from typing import Callable

from src.imgdataconvertcodegen.convert_code_generation import ConvertCodeGenerator
from src.imgdataconvertcodegen.knowledge_graph_construction.get_knowledge_graph_builder import (
    get_knowledge_graph_builder)

_builder = get_knowledge_graph_builder()
_code_generator = ConvertCodeGenerator(_builder.knowledge_graph)


def get_covert_code(source_var_name: str, source_spec: str | dict, target_var_name: str, target_spec: str | dict):
    """
    Generates Python code as a string that performs data conversion from a source variable to a target variable
    Args:
        source_var_name: the name of the variable holding the source data.
        source_spec: the name of library or a dictionary containing metadata about the source data.
        target_var_name:  the name of the variable that will store the result of the conversion.
        target_spec: the same as source_spec

    Returns: A string containing the Python code necessary to perform the conversion.

    """
    return _code_generator.generate_code(source_var_name, source_spec, target_var_name, target_spec)


def get_convert_path(source_spec: str | dict, target_spec: str | dict):
    """
    get the convert path from source to target
    Args:
        source_spec: the name of library or a dictionary containing metadata about the source data.
        target_spec: the same as source_spec

    Returns: A list of metadata

    """
    return _code_generator.get_convert_path(source_spec, target_spec)


def add_image_metadata(new_metadata):
    _builder.add_new_metadata(new_metadata)
    _code_generator.knowledge_graph = _builder.knowledge_graph


def add_convert_code_factory(new_factory: Callable | str):
    _builder.add_new_edge_factory(new_factory)
    _code_generator.knowledge_graph = _builder.knowledge_graph


def add_new_lib_preset(lib_name, metadata):
    _builder.add_lib_preset(lib_name, metadata)
    _code_generator.knowledge_graph = _builder.knowledge_graph
