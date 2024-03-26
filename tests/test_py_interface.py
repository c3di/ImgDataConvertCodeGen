import pytest

from imgdataconvertcodegen import add_conversion_for_metadata_pairs, _code_generator, _constructor


@pytest.fixture
def conversion_for_metadata_pairs():
    return [({"color_channel": "bgr", "channel_order": "channel last", "minibatch_input": True,
              "image_data_type": "uint8", "device": "gpu", "data_representation": "torch.tensor"},
             {"color_channel": "rgb", "channel_order": "channel first", "minibatch_input": True,
              "image_data_type": "uint8", "device": "gpu", "data_representation": "torch.tensor"},
             ("", "def convert(var)\n  return var[:, :, ::-1]")),
            ({"color_channel": "rgb", "channel_order": "channel first", "minibatch_input": True,
              "image_data_type": "uint8", "device": "gpu", "data_representation": "torch.tensor"},
             {"color_channel": "bgr", "channel_order": "channel last", "minibatch_input": True,
              "image_data_type": "uint8", "device": "gpu", "data_representation": "torch.tensor"},
             ("", "def convert(var)\n  return var[:, :, ::-1]"))
            ]


def test_add_conversion_for_metadata_pair_single_value(conversion_for_metadata_pairs):
    _constructor.clear_knowledge_graph()
    pair = conversion_for_metadata_pairs[0]
    add_conversion_for_metadata_pairs(pair)
    assert _code_generator.knowledge_graph.nodes == [pair[0], pair[1]]
    assert _code_generator.knowledge_graph.edges == [(pair[0], pair[1])]
    edge_data = _code_generator.knowledge_graph.get_edge_data(pair[0], pair[1])
    edge_data["conversion"] == pair[2]
    edge_data["factory"] == "manual"


def test_add_conversion_for_metadata_pair_list_values(conversion_for_metadata_pairs):
    _constructor.clear_knowledge_graph()
    add_conversion_for_metadata_pairs(conversion_for_metadata_pairs)
    for pair in conversion_for_metadata_pairs:
        assert pair[0] in _code_generator.knowledge_graph.nodes
        assert pair[1] in _code_generator.knowledge_graph.nodes
        assert (pair[0], pair[1]) in _code_generator.knowledge_graph.edges
        edge_data = _code_generator.knowledge_graph.get_edge_data(pair[0], pair[1])
        edge_data["conversion"] == pair[2]
        edge_data["factory"] == "manual"


def test_add_conversion_for_metadata_pair_empty():
    _constructor.clear_knowledge_graph()
    add_conversion_for_metadata_pairs([])
    assert _code_generator.knowledge_graph.nodes == []
    assert _code_generator.knowledge_graph.edges == []


def test_add_conversion_for_metadata_pair_none():
    _constructor.clear_knowledge_graph()
    add_conversion_for_metadata_pairs(None)
    assert _code_generator.knowledge_graph.nodes == []
    assert _code_generator.knowledge_graph.edges == []
