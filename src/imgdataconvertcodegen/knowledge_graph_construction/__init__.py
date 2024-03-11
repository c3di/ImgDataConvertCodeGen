from .edge_factories import all_edge_factories
from .knowledge_graph import KnowledgeGraph
from .constructor import KnowledgeGraphConstructor
from .metedata import *


constructor = KnowledgeGraphConstructor(img_metadata_config, all_edge_factories)
constructor.build()


def get_knowledge_graph_constructor():
    return constructor
