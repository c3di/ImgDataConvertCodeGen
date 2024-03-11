from typing import TypedDict, Literal


class MetadataValues(TypedDict):
    data_representation: list[str]
    color_channel: list[str]
    channel_order: list[Literal['channel last', 'channel first', 'none']]
    minibatch_input: list[bool]
    data_type: list[str]
    intensity_range: list[Literal['full', 'normalized_unsigned', 'normalized_signed']]
    device: list[str]


metadata_values: MetadataValues = {
    "data_representation": ["torch.tensor", "numpy.ndarray", "PIL.Image", "tf.tensor"],
    "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
    "channel_order": ['channel last', 'channel first', 'none'],
    "minibatch_input": [True, False],
    "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64', 'double',
                  'int8', 'int16', 'int32', 'int64'],
    # normalized_unsigned: 0-1, normalized_signed: -1 to 1
    "intensity_range": ['full', 'normalized_unsigned', 'normalized_signed'],
    "device": ['cpu', 'gpu']
}


# todo: to remove
def is_valid_metadata(metadata: dict):
    if metadata['color_channel'] == 'rgba' or metadata['color_channel'] == 'graya':
        if metadata['data_representation'] == 'PIL.Image':
            return metadata['data_type'] == 'uint8'
        return False
    if metadata['data_representation'] == 'numpy.ndarray':
        return metadata['minibatch_input'] == False
    if metadata['data_representation'] == 'torch.tensor':
        for type in ['uint8', 'float32', 'int8', 'int16', 'int32', 'float64', 'double', 'int64']:
            if metadata['data_type'] == type:
                if metadata['data_type'] == 'float32' and metadata['intensity_range'] != 'normalized_unsigned':
                    return False
                return True
        return False
    return True


class Metadata(TypedDict):
    data_representation: str
    color_channel: str
    channel_order: Literal['channel last', 'channel first', 'none']
    minibatch_input: bool
    data_type: Literal[
        'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'double', 'int8', 'int16', 'int32', 'int64']
    intensity_range: Literal['full', 'normalized_unsigned', 'normalized_signed']
    device: str
