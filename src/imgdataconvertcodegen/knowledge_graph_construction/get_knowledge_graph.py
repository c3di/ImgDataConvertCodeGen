from .default_edges_factories import default_factories
from .metadata_values import metadata_values
from .lib_presets import lib_presets
from .knowledge_graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(metadata_values, default_factories, lib_presets)
knowledge_graph = builder.build()


def get_knowledge_graph():
    return knowledge_graph
