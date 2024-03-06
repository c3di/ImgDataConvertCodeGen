import itertools
import os.path
from typing import Callable

from .knowledge_graph import KnowledgeGraph
from .metadata import ValuesOfImgRepr, ImageRepr, is_valid_value_of_img_repr
from ..util import exclude_key_from_list


class KnowledgeGraphBuilder:
    def __init__(self, bunch_of_img_repr: dict[ImageRepr, ValuesOfImgRepr], edge_factories: list[Callable] = []):
        self._bunch_of_img_repr = bunch_of_img_repr
        self._edge_factories = edge_factories
        self._know_graph_file_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
        self._graph = KnowledgeGraph()

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
        return self.knowledge_graph

    def build_from_file(self, path):
        self.knowledge_graph.load_from_file(path)

    def build_from_scratch(self, factories_to_use: list[Callable]):
        all_img_repr = list(self._bunch_of_img_repr.keys())
        for img_repr in all_img_repr:
            self._build_from_repr(img_repr, self._bunch_of_img_repr[img_repr], factories_to_use)
        self.save_knowledge_graph()

    def _build_from_repr(self, img_repr, values_for_repr, factories_to_use: list[Callable]):
        values_for_repr['data_representation'] = [img_repr]
        # one property change policy
        keys = list(values_for_repr.keys())
        for source_value in itertools.product(*list(values_for_repr.values())):
            source = dict(zip(keys, source_value))
            for attribute in values_for_repr.keys():
                for target_value in exclude_key_from_list(values_for_repr[attribute], source[attribute]):
                    self._create_edge_in_kg(source, attribute, target_value, factories_to_use)

            for another_img_repr in exclude_key_from_list(self._bunch_of_img_repr.keys(), img_repr):
                if not is_valid_value_of_img_repr(source, self._bunch_of_img_repr[another_img_repr]):
                    continue
                self._create_edge_in_kg(source, 'data_representation', another_img_repr, factories_to_use)

    def _create_edge_in_kg(self, source, changed_attribute: str, new_value, factories_to_use: list[Callable] = []):
        target = source.copy()
        target[changed_attribute] = new_value
        for factory in factories_to_use:
            function = factory(source, target)
            if function is not None:
                used_factory = f'{factory.__code__.co_name} in {factory.__code__.co_filename}'
                self.knowledge_graph.add_edge(source, target,
                                              conversion=function, factory=used_factory)

    def add_new_edge_factory(self, factory: Callable):
        self._edge_factories.append(factory)
        self.build_from_scratch([factory])

    def add_img_repr(self, img_repr: str, values_for_repr: ValuesOfImgRepr):
        self._build_from_repr(img_repr, values_for_repr, self._edge_factories)
        self.save_knowledge_graph()
