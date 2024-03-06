import os

from .edge_factories import all_edge_factories
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .metadata import bunch_of_img_repr

dev_mode = os.getenv('Dev')

builder = KnowledgeGraphBuilder(bunch_of_img_repr, all_edge_factories)
builder.build(dev_mode)


def get_knowledge_graph_builder():
    return builder
