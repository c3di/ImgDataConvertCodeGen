import pytest

from imgdataconvertcodegen.knowledge_graph_construction.metadata import is_valid_metadata_pair, \
    check_metadata_value_valid, encode_metadata, decode_metadata, find_closest_metadata


def test_is_valid_metadata_pair():
    assert is_valid_metadata_pair({'key1': 'value1'}, {'key1': 'value2'}) is True
    assert is_valid_metadata_pair({'key1': 'value1'}, {'key1': 'value1'}) is False


def test_assert_metadata_value_valid():
    metadata = {
        "data_representation": "torch.tensor",
        "color_channel": 'rgb',
        "channel_order": 'channel first',
        "minibatch_input": True,
        "data_type": 'uint8',
        "intensity_range": 'full',
        "device": 'gpu'
    }
    check_metadata_value_valid(metadata)
    with pytest.raises(ValueError) as excinfo:
        incorrect_metadata = metadata.copy()
        incorrect_metadata['color_channel'] = 'invalid_value'
        check_metadata_value_valid(incorrect_metadata)
    assert 'Invalid metadata' in str(excinfo.value)


def test_encode_to_string():
    metadata = {
        "data_representation": "torch.tensor",
        "color_channel": 'rgb',
        "channel_order": 'channel first',
        "minibatch_input": True,
        "data_type": 'uint8',
        "intensity_range": 'full',
        "device": 'gpu'
    }
    encoded = encode_metadata(metadata)
    assert encoded == 'torch.tensor-rgb-channel first-True-uint8-full-gpu'


def test_decode_to_dict():
    metadata_str = 'torch.tensor-rgb-channel first-True-uint8-full-gpu'
    decoded = decode_metadata(metadata_str)
    assert decoded == {
        "data_representation": "torch.tensor",
        "color_channel": 'rgb',
        "channel_order": 'channel first',
        "minibatch_input": True,
        "data_type": 'uint8',
        "intensity_range": 'full',
        "device": 'gpu'
    }


source_metadata = {
    "data_representation": "torch.tensor",
    "color_channel": "rgb",
    # Other metadata values...
}


@pytest.fixture
def candidate_metadatas():
    return [
        {"data_representation": "torch.tensor", "color_channel": "rgb"},
        {"data_representation": "numpy.ndarray", "color_channel": "rgb"},
        {"data_representation": "torch.tensor", "color_channel": "bgr"},
        {"data_representation": "tf.tensor", "color_channel": "rgba"},
    ]


def test_exact_match(candidate_metadatas):
    closest_targets = find_closest_metadata(source_metadata, candidate_metadatas)
    assert closest_targets == candidate_metadatas[0]


def test_representation_match_no_channel_match(candidate_metadatas):
    modified_source = source_metadata.copy()
    modified_source["color_channel"] = "rgba"
    closest_targets = find_closest_metadata(modified_source, candidate_metadatas)
    assert closest_targets == candidate_metadatas[0]


def test_no_representation_match(candidate_metadatas):
    modified_source = source_metadata.copy()
    modified_source["data_representation"] = "unknown"
    closest_targets = find_closest_metadata(modified_source, candidate_metadatas)
    assert closest_targets == candidate_metadatas[0]


def test_rgb_bgr_match(candidate_metadatas):
    modified_source = source_metadata.copy()
    modified_source["color_channel"] = "bgr"
    modified_source["data_representation"] = "numpy.ndarray"
    closest_targets = find_closest_metadata(modified_source, candidate_metadatas)
    assert closest_targets == candidate_metadatas[1]


def test_empty_candidate_list():
    closest_targets = find_closest_metadata(source_metadata, [])
    assert closest_targets is None


def test_one_candidata():
    closest_targets = find_closest_metadata(source_metadata, [source_metadata])
    assert closest_targets == source_metadata
