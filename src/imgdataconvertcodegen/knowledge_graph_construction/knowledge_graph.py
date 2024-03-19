import networkx as nx

from .io import save_graph, load_graph
from .metedata import Metadata, encode_metadata, decode_metadata


class KnowledgeGraph:
    _graph = None

    def __init__(self):
        self._graph = nx.DiGraph()

    def clear(self):
        self._graph.clear()

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        encoded_edges = self._graph.edges
        return [(decode_metadata(edge[0]), decode_metadata(edge[1])) for edge in encoded_edges]

    def add_node(self, node):
        self._graph.add_node(encode_metadata(node))

    def add_edge(self, source: Metadata, target: Metadata, conversion, factory=None):
        self._graph.add_edge(encode_metadata(source), encode_metadata(target), conversion=conversion, factory=factory)

    def get_edge_data(self, source: Metadata, target: Metadata):
        return self._graph.get_edge_data(encode_metadata(source), encode_metadata(target))

    def save_to_file(self, path):
        save_graph(self._graph, path)

    def load_from_file(self, path):
        self._graph = load_graph(path)

    def heuristic(self, u, v):
        # todo: how to design the cost function for edge and the heuristic function for each node
        return 0

    def get_shortest_path(self, source_metadata, target_metadata) -> list[str] | None:
        try:
            path = nx.astar_path(self._graph, encode_metadata(source_metadata), encode_metadata(target_metadata),
                                 heuristic=self.heuristic)
            return [decode_metadata(node) for node in path]
        except nx.NetworkXNoPath:
            return None
        return path

    def __str__(self):
        return f"Knowledge Graph with {len(self._graph)} nodes and {len(self._graph.edges)} edges."
