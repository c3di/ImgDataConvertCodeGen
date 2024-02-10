metadata_values = {
    "data_representation": ["torch.tensor", "numpy.ndarray", "PIL.Image", "tf.tensor"],
    "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
    "channel_order": ['channel last', 'channel first', 'none'],
    "minibatch_input": [True, False],
    "data_type": ['uint8', 'uint16', 'uint32', 'float32', 'int8', 'int16', 'int32', 'int64', 'double'],
    "intensity_range": ['0to255', '0to1', '-1to1'],
    "device": ['cpu', 'gpu']
}


def check_metadata_valid(metadata: dict):
    for key, value_list in metadata_values.items():
        if not metadata.get(key) in value_list:
            raise f'Invalid metadata: {metadata} at key: {key}. expected value list: {value_list}'
