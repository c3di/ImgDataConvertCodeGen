import networkx as nx
from ..io import save_graph


class KnowledgeGraph:
    _graph = None
    _uuid = 1

    def __init__(self, nodes, edge_factories):
        self._graph = nx.DiGraph()
        self._edge_factories = edge_factories
        self._build_graph(nodes, self._edge_factories)

    def add_node(self, node):
        self._graph.add_node(self._uuid, **node)
        self._uuid += 1

    def get_node_id(self, node):
        for node_id in self._graph.nodes:
            if self._graph.nodes[node_id] == node:
                return node_id

    def get_node(self, node_id):
        return self._graph.nodes[node_id]

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, source_id, target_id, conversion):
        self._graph.add_edge(source_id, target_id, conversion=conversion)

    def get_edge(self, source_id, target_id):
        return self._graph.get_edge_data(source_id, target_id)

    def add_new_edge_factory(self, factory):
        self._edge_factories.append(factory)
        self._create_edges(self._edge_factories)

    def save_to_file(self, path):
        save_graph(self._graph, path)

    def heuristic(self, u, v):
        # todo: how to design the cost function for edge and the heuristic function for each node
        return 0

    def get_shortest_path(self, source_metadata, target_metadata) -> list[str] | None:
        return self._get_shortest_path_using_id(self.get_node_id(source_metadata), self.get_node_id(target_metadata))

    def _get_shortest_path_using_id(self, source_id, target_id) -> list[str] | None:
        try:
            path = nx.astar_path(self._graph, source_id, target_id,
                                 heuristic=self.heuristic)
        except nx.NetworkXNoPath:
            return None
        return path

    def _build_graph(self, nodes, edge_factories):
        self.add_nodes(nodes)
        self._create_edges(edge_factories)

    def _create_edges(self, edge_factories):
        for source_id in self._graph:
            source = self._graph.nodes[source_id]
            for target_id in self._graph:
                target = self._graph.nodes[target_id]
                if source != target:
                    for factory in edge_factories:
                        function = factory(source, target)
                        if function is not None:
                            self.add_edge(source_id, target_id, conversion=function)
                            break

    def __str__(self):
        return f"Knowledge Graph with {len(self._graph)} nodes and {len(self._graph.edges)} edges."
