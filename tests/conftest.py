import pytest
from imgdataconvertcodegen.knowledge_graph_construction import get_knowledge_graph_builder


@pytest.fixture(scope="session", autouse=True)
def force_to_rebuild_kg_for_tests():
    builder = get_knowledge_graph_builder()
    builder.build_from_scratch(builder._edge_factories)
    print("Knowledge graph rebuilt for tests")
