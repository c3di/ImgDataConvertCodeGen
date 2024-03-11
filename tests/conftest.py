import pytest
from imgdataconvertcodegen.knowledge_graph_construction import get_knowledge_graph_constructor


@pytest.fixture(scope="session", autouse=True)
def force_to_rebuild_kg_for_tests():
    constructor = get_knowledge_graph_constructor()
    constructor.build_from_scratch(constructor._edge_factories)
