from typing import TypedDict, Literal


class MetadataValues(TypedDict):
    data_representation: list[str]
    color_channel: list[str]
    channel_order: list[Literal['channel last', 'channel first', 'none']]
    minibatch_input: list[bool]
    data_type: list[str]
    intensity_range: list[Literal['full', '0to1', '-1to1']]
    device: list[str]


metadata_values: MetadataValues = {
    "data_representation": ["torch.tensor", "numpy.ndarray", "PIL.Image", "tf.tensor"],
    "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
    "channel_order": ['channel last', 'channel first', 'none'],
    "minibatch_input": [True, False],
    "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
                  'float16', 'float32', 'float64', 'double',
                  'int8', 'int16', 'int32', 'int64'],
    "intensity_range": ['full', '0to1', '-1to1'],
    "device": ['cpu', 'gpu']
}


class Metadata(TypedDict):
    data_representation: str
    color_channel: str
    channel_order: Literal['channel last', 'channel first', 'none']
    minibatch_input: bool
    data_type: Literal[
        'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'double', 'int8', 'int16', 'int32', 'int64']
    intensity_range: Literal['full', '-1to1', '0to1']
    device: str
