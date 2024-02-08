from .knowledge_graph import KnowledgeGraph
from .default_edges_factories import default_factories
from .default_nodes import default_nodes


def create_knowledge_graph():
    return KnowledgeGraph(default_nodes, default_factories)
