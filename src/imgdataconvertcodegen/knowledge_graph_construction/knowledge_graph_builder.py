import itertools
import os.path
from datetime import datetime
from typing import Callable

from .knowledge_graph import KnowledgeGraph
from .metadata_values import assert_metadata_valid
from ..metadata_differ import is_same_metadata
from ..measure import get_execution_time


class KnowledgeGraphBuilder:
    def __init__(self, metadata_values, edge_factories, lib_presets):
        self._metadata_values = metadata_values
        self._edge_factories = edge_factories
        self._know_graph_file_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
        self._graph = KnowledgeGraph(lib_presets)

    @property
    def knowledge_graph(self):
        return self._graph

    def save_knowledge_graph(self):
        self.knowledge_graph.save_to_file(self._know_graph_file_path)

    def build(self, force_to_rebuild=False) -> KnowledgeGraph:
        if not force_to_rebuild and os.path.exists(self._know_graph_file_path):
            self.build_from_file(self._know_graph_file_path)
        else:
            self.build_from_scratch(self._edge_factories)
        print(self.knowledge_graph)
        return self.knowledge_graph

    def build_from_file(self, path):
        start_time = datetime.now()
        self.knowledge_graph.load_from_file(path)
        end_time = datetime.now()
        print(get_execution_time(start_time, end_time))

    def build_from_scratch(self, factories_to_use: list[Callable]):
        start_time = datetime.now()
        keys = list(self._metadata_values.keys())
        values_lists = list(self._metadata_values.values())
        for source_value in itertools.product(*values_lists):
            source_metadata = dict(zip(keys, source_value))
            self._build_for_metadata(source_metadata, factories_to_use)
        end_time = datetime.now()
        print(get_execution_time(start_time, end_time))
        self.save_knowledge_graph()

    def _build_for_metadata(self, source_metadata, factories_to_use: list[Callable]):
        source_id = None
        for attribute_name in self._metadata_values.keys():
            for target_value in self._metadata_values[attribute_name]:
                target_metadata = source_metadata.copy()
                target_metadata[attribute_name] = target_value
                if is_same_metadata(source_metadata, target_metadata):
                    continue
                convert_function = self._create_conversion_function(source_metadata, target_metadata, factories_to_use)
                if convert_function is not None:
                    if source_id is None:
                        source_id = self.knowledge_graph.add_node(source_metadata)
                    target_id = self.knowledge_graph.add_node(target_metadata)
                    self.knowledge_graph.add_edge(source_id, target_id, conversion=convert_function)

    def _create_conversion_function(self, source, target, factories_to_use: list[Callable] = []):
        for factory in factories_to_use:
            function = factory(source, target)
            if function is not None:
                return function
        return None

    def add_new_edge_factory(self, factory: Callable):
        self._edge_factories.append(factory)
        self.build_from_scratch([factory])

    def add_new_metadata(self, new_metadata: dict):
        assert_metadata_valid(new_metadata)
        if self.knowledge_graph.is_node_exist(new_metadata):
            return
        self._build_for_metadata(new_metadata, self._edge_factories)
        self.save_knowledge_graph()

    def add_lib_preset(self, lib_name: str, metadata: dict):
        self.knowledge_graph.add_lib_preset(lib_name, metadata)
