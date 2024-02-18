from src.imgdataconvertcodegen.knowledge_graph_construction.edge_factories import all_edge_factories
from .metadata_values import metadata_values
from .lib_presets import lib_presets
from .knowledge_graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(metadata_values, all_edge_factories, lib_presets)
builder.build()


def get_knowledge_graph_builder():
    return builder
