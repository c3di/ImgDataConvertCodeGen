import networkx as nx
from .nodes import is_single_metadata_differ, nodes
from .edges_factories import factories
from ..io import save_graph

conversion_graph_path = 'conversion_knowledge.json'


def build_graph(save_path=conversion_graph_path):
    graph = nx.DiGraph()

    def add_nodes():
        idx = 0
        for node in nodes:
            graph.add_node(idx, **node)
            idx += 1

    def add_edges():
        for source_id in graph:
            source = graph.nodes[source_id]
            for target_id in graph:
                target = graph.nodes[target_id]
                if is_single_metadata_differ(source, target):
                    for factory in factories:
                        routine = factory(source, target)
                        if routine is not None:
                            graph.add_edge(source_id, target_id,
                                           routine=routine)
                            break

    add_nodes()
    add_edges()
    save_graph(graph, save_path)
