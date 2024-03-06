import networkx as nx

from .io import save_graph, load_graph
from .metadata import check_metadata_value_valid


class KnowledgeGraph:
    _graph = None
    _uuid = 0

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, node) -> int:
        node_id = self.get_node_id(node)
        if node_id is not None:
            return node_id
        self._uuid += 1
        self._graph.add_node(self._uuid, **node)
        return self._uuid

    def is_node_exist(self, node):
        return self.get_node_id(node) is not None

    def get_node_id(self, node):
        for node_id in self._graph.nodes:
            if self._graph.nodes[node_id] == node:
                return node_id
        return None

    def get_node(self, node_id):
        return self._graph.nodes[node_id]

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    def add_edge(self, source_id, target_id, conversion, factory=None):
        self._graph.add_edge(source_id, target_id, conversion=conversion, factory=factory)

    def get_edge_data(self, source_id, target_id):
        return self._graph.get_edge_data(source_id, target_id)

    def save_to_file(self, path):
        save_graph(self._graph, path)

    def load_from_file(self, path):
        self._graph = load_graph(path)
        self._uuid = max(self._graph.nodes)

    def heuristic(self, u, v):
        # todo: how to design the cost function for edge and the heuristic function for each node
        return 0

    def get_shortest_path(self, source_metadata, target_metadata) -> list[str] | None:
        """
        Returns: the id list of the shortest path

        """
        return self._get_shortest_path_using_id(self.get_node_id(source_metadata), self.get_node_id(target_metadata))

    def _get_shortest_path_using_id(self, source_id, target_id) -> list[str] | None:
        try:
            path = nx.astar_path(self._graph, source_id, target_id,
                                 heuristic=self.heuristic)
        except nx.NetworkXNoPath:
            return None
        return path

    def __str__(self):
        return f"Knowledge Graph with {len(self._graph)} nodes and {len(self._graph.edges)} edges."
