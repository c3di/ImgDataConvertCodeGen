import itertools
import os.path
from typing import Callable

from .knowledge_graph import KnowledgeGraph
from .metedata import MetadataValues
from ..util import exclude_key_from_list


class KnowledgeGraphConstructor:
    def __init__(self, metadata_values: MetadataValues, edge_factories: list[Callable] = []):
        self._metadata_values = metadata_values
        self._edge_factories = edge_factories
        self._know_graph_file_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
        self._graph = KnowledgeGraph()

    @property
    def knowledge_graph(self):
        return self._graph

    def save_knowledge_graph(self):
        self.knowledge_graph.save_to_file(self._know_graph_file_path)

    def build(self) -> KnowledgeGraph:
        if os.path.exists(self._know_graph_file_path):
            self.build_from_file(self._know_graph_file_path)
        else:
            self.build_from_scratch()
        return self.knowledge_graph

    def build_from_file(self, path):
        self.knowledge_graph.load_from_file(path)

    def build_from_scratch(self):
        self._create_edges_from_metadata_values(self._metadata_values, self._edge_factories)

    def _create_edges_from_metadata_values(self, metadata_values, factories_to_use: list[Callable]):
        attributes = list(metadata_values.keys())
        attributes_values_lists = list(metadata_values.values())
        for attribute_values in itertools.product(*attributes_values_lists):
            source_metadata = dict(zip(attributes, attribute_values))
            self._create_edges_that_start_from(source_metadata, metadata_values, factories_to_use)
        self.save_knowledge_graph()

    def _create_edges_that_start_from(self, source_metadata, metadata_values, factories_to_use: list[Callable]):
        for attribute in metadata_values.keys():
            for new_attribute_v in exclude_key_from_list(metadata_values[attribute], source_metadata[attribute]):
                self._create_edge(source_metadata, attribute, new_attribute_v, factories_to_use)

    def _create_edge(self, source, changed_attribute: str, new_attribute_value,
                     factories_to_use: list[Callable] = []):
        """
        one property change policy.
        """
        target = source.copy()
        target[changed_attribute] = new_attribute_value
        for factory in factories_to_use:
            function = factory(source, target)
            if function is not None:
                used_factory = f'{factory.__code__.co_name} in {factory.__code__.co_filename}'
                self.knowledge_graph.add_edge(source, target,
                                              conversion=function, factory=used_factory)

    def add_new_edge_factory(self, factory: Callable):
        self._edge_factories.append(factory)
        self._create_edges_from_metadata_values(self._metadata_values, [factory])

    def add_new_metadata_values(self, new_metadata: MetadataValues):
        self._create_edges_from_metadata_values(new_metadata, self._edge_factories)
