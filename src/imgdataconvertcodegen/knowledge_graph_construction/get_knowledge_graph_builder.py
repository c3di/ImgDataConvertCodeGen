import os

from src.imgdataconvertcodegen.knowledge_graph_construction.edge_factories import all_edge_factories
from .metadata_values import metadata_values
from .lib_presets import lib_presets
from .knowledge_graph_builder import KnowledgeGraphBuilder

development_mode = os.getenv('Development')

builder = KnowledgeGraphBuilder(metadata_values, all_edge_factories, lib_presets)
#todo add a way to set development mode from outside
builder.build(development_mode is None)


def get_knowledge_graph_builder():
    return builder
