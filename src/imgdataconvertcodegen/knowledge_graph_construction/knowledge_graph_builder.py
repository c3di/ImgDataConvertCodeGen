import itertools
import os.path
import warnings

from .knowledge_graph import KnowledgeGraph
from ..metadata_differ import is_single_metadata_differ


class KnowledgeGraphBuilder:
    def __init__(self, metadata_values, edge_factories, lib_presets):
        self._metadata_values = metadata_values
        self._edge_factories = edge_factories
        self._know_graph_file_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
        self._graph = KnowledgeGraph(lib_presets)

    @property
    def knowledge_graph(self):
        return self._graph

    def build(self) -> KnowledgeGraph:
        if os.path.exists(self._know_graph_file_path):
            self.build_from_file(self._know_graph_file_path)
        else:
            warnings.warn("The knowledge graph file does not exist. The system will construct the knowledge graph"
                          " from scratch, which may take some time..")
            self.build_from_scratch()
        print(self.knowledge_graph)
        return self.knowledge_graph

    def build_from_file(self, path):
        self.knowledge_graph.load_from_file(path)

    def build_from_scratch(self):
        keys = list(self._metadata_values.keys())
        values_lists = list(self._metadata_values.values())
        for source_value in itertools.product(*values_lists):
            source_metadata = dict(zip(keys, source_value))
            for target_value in itertools.product(*values_lists):
                target_metadata = dict(zip(keys, target_value))
                allow_edge_exist = is_single_metadata_differ(source_metadata, target_metadata)
                if allow_edge_exist:
                    convert_function = self._create_conversion_function(source_metadata, target_metadata)
                    if convert_function is not None:
                        source_id = self.knowledge_graph.add_node(source_metadata)
                        target_id = self.knowledge_graph.add_node(target_metadata)
                        self.knowledge_graph.add_edge(source_id, target_id, conversion=convert_function)
        self.knowledge_graph.save_to_file(self._know_graph_file_path)

    def _create_conversion_function(self, source, target):
        for factory in self._edge_factories:
            function = factory(source, target)
            if function is not None:
                return function
        return None

    def add_new_edge_factory(self, factory):
        self._edge_factories.append(factory)
        self.build_from_scratch()

    def __str__(self):
        # todo: print the graph
        return "KnowledgeGraphBuilder"
