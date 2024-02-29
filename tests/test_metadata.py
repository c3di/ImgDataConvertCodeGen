import pytest

from imgdataconvertcodegen.knowledge_graph_construction.metadata import is_valid_metadata_pair, \
    check_metadata_value_valid, encode_metadata, decode_metadata


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
