"""
This module is used to construct the knowledge graph for the image data conversion.
The knowledge graph is a directed graph, where each node represents a metadata of an image, and each edge represents a conversion from one metadata to another metadata.

To create an edge in the graph,  it's essential to ensure that:
    * Each attribute adheres to the valid values specified by the libraries.
    * The combination of attribute values forms a valid metadata entity.
    * There is a feasible conversion code between the source and target metadata.
"""
import itertools
import os.path
from typing import Callable

from .knowledge_graph import KnowledgeGraph
from .metedata import PossibleValuesForImgRepr, ImgMetadataConfigDict, is_valid_attribute_value, ImgRepr, \
    ImgMetadataConfig, ValidCheckFunc
from ..util import exclude_key_from_list


class KnowledgeGraphConstructor:
    def __init__(self, img_metadata_config_dict: ImgMetadataConfigDict, edge_factories: list[Callable] = []):
        self._img_metadata_config_dict = img_metadata_config_dict
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
            self.build_from_scratch(self._edge_factories)
        return self.knowledge_graph

    def build_from_file(self, path):
        self.knowledge_graph.load_from_file(path)

    def build_from_scratch(self, factories_to_use: list[Callable]):
        all_img_repr = list(self._img_metadata_config_dict.keys())
        for img_repr in all_img_repr:
            self._create_edges_from_metadata_config(img_repr, self._img_metadata_config_dict[img_repr], factories_to_use)
        self.save_knowledge_graph()

    def _create_edges_from_metadata_config(self, img_repr: ImgRepr, config: ImgMetadataConfig, factories_to_use: list[Callable]):
        possible_values, metadata_valid_check = config
        keys = list(possible_values.keys())
        for source_value in itertools.product(*list(possible_values.values())):
            source = dict(zip(keys, source_value))
            source['data_representation'] = img_repr
            self._create_edges_that_start_from(source, metadata_valid_check, possible_values, factories_to_use)

    def _create_edges_that_start_from(self, source, valid_check: ValidCheckFunc, possible_values,
                                      factories_to_use: list[Callable]):
        if not valid_check(source):
            return
        for changed_attribute in possible_values.keys():
            for new_attribute_value in exclude_key_from_list(possible_values[changed_attribute],
                                                             source[changed_attribute]):
                self._create_edge(source, changed_attribute, new_attribute_value, valid_check, factories_to_use)

        for another_img_repr in exclude_key_from_list(self._img_metadata_config_dict.keys(),
                                                      source['data_representation']):
            possible_values, valid_check = self._img_metadata_config_dict[another_img_repr]
            if not is_valid_attribute_value(source, possible_values):
                continue
            self._create_edge(source, 'data_representation', another_img_repr, valid_check, factories_to_use)

    def _create_edge(self, source, changed_attribute: str, new_attribute_value, valid_check: ValidCheckFunc,
                     factories_to_use: list[Callable] = []):
        """
        one property change policy.
        """
        target = source.copy()
        target[changed_attribute] = new_attribute_value
        if not valid_check(target):
            return
        for factory in factories_to_use:
            function = factory(source, target)
            if function is not None:
                used_factory = f'{factory.__code__.co_name} in {factory.__code__.co_filename}'
                self.knowledge_graph.add_edge(source, target,
                                              conversion=function, factory=used_factory)

    def add_new_edge_factory(self, factory: Callable):
        self._edge_factories.append(factory)
        self.build_from_scratch([factory])

    def add_new_img_metadata_config(self, img_repr: str, config: ImgMetadataConfig):
        self._create_edges_from_metadata_config(img_repr, config, self._edge_factories)
        self.save_knowledge_graph()
