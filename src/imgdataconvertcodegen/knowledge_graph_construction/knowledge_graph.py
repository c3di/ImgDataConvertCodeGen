import networkx as nx
from ..io import save_graph, load_graph


class KnowledgeGraph:
    _graph = None
    _uuid = 0

    def __init__(self, lib_presets=None):
        self._graph = nx.DiGraph()
        self._lib_presets = lib_presets

    def add_node(self, node) -> int:
        node_id = self.get_node_id(node)
        if node_id is not None:
            return node_id
        self._uuid += 1
        self._graph.add_node(self._uuid, **node)
        return self._uuid

    def get_node_id(self, node):
        for node_id in self._graph.nodes:
            if self._graph.nodes[node_id] == node:
                return node_id
        return None

    def get_node(self, node_id):
        return self._graph.nodes[node_id]

    def add_edge(self, source_id, target_id, conversion):
        self._graph.add_edge(source_id, target_id, conversion=conversion)

    def get_edge(self, source_id, target_id):
        return self._graph.get_edge_data(source_id, target_id)

    def save_to_file(self, path):
        save_graph(self._graph, path)
        print(f"Knowledge Graph has been saved to {path}")

    def load_from_file(self, path):
        self._graph = load_graph(path)
        print(f"Knowledge Graph has been loaded from {path}")

    def heuristic(self, u, v):
        # todo: how to design the cost function for edge and the heuristic function for each node
        return 0

    def get_shortest_path(self, source_metadata, target_metadata) -> list[str] | None:
        return self._get_shortest_path_using_id(self.get_node_id(source_metadata), self.get_node_id(target_metadata))

    def get_metadata_by_lib_name(self, lib_name):
        if lib_name not in self._lib_presets:
            raise ValueError(f"{lib_name} not in the lib_presets. "
                             f"We support {list(self._lib_presets.keys())}")
        return self._lib_presets[lib_name]

    def _get_shortest_path_using_id(self, source_id, target_id) -> list[str] | None:
        try:
            path = nx.astar_path(self._graph, source_id, target_id,
                                 heuristic=self.heuristic)
        except nx.NetworkXNoPath:
            return None
        return path

    # def _build_graph(self, metadata_values):
    #     keys = list(metadata_values.keys())
    #     values_lists = list(metadata_values.values())
    #     i = 0
    #     for source_value in itertools.product(*values_lists):
    #         source_metadata = dict(zip(keys, source_value))
    #         for target_value in itertools.product(*values_lists):
    #             target_metadata = dict(zip(keys, target_value))
    #             allow_edge_exist = is_single_metadata_differ(source_metadata, target_metadata)
    #             if allow_edge_exist:
    #                 convert_function = self._create_conversion_function(source_metadata, target_metadata)
    #                 if convert_function is not None:
    #                     source_id = self.add_node(source_metadata)
    #                     target_id = self.add_node(target_metadata)
    #                     self.add_edge(source_id, target_id, conversion=convert_function)
    #
    # def _create_conversion_function(self, source, target):
    #     for factory in self._edge_factories:
    #         function = factory(source, target)
    #         if function is not None:
    #             return function
    #     return None

    def __str__(self):
        return f"Knowledge Graph with {len(self._graph)} nodes and {len(self._graph.edges)} edges."
