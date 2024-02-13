metadata_values = {
    "data_representation": ["torch.tensor", "numpy.ndarray", "PIL.Image", "tf.tensor"],
    "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
    "channel_order": ['channel last', 'channel first', 'none'],
    "minibatch_input": [True, False],
    "data_type": ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'int8',
                  'float64', 'double', 'int16', 'int32', 'int64'],
    # normalized_unsigned: 0-1, normalized_signed: -1 to 1
    "intensity_range": ['full', 'normalized_unsigned', 'normalized_signed'],
    "device": ['cpu', 'gpu']
}


def assert_metadata_valid(metadata: dict):
    for key, value_list in metadata_values.items():
        if not metadata.get(key) in value_list:
            assert f'Invalid metadata: {metadata} at key: {key}. expected value list: {value_list}'


def encode_to_string(metadata: dict) -> str:
    return '-'.join([str(v) for v in metadata.values()])


def decode_to_dict(metadata_str: str) -> dict:
    metadata_list = metadata_str.split('-')
    return {k: v for k, v in zip(metadata_values.keys(), metadata_list)}
