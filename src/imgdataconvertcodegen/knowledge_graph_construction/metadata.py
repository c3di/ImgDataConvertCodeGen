from typing import TypedDict, Literal

# todo: to remove
metadata_values = {
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


ImageRepr = str


class ValuesOfImgRepr(TypedDict):
    color_channel: list[str]
    channel_order: list[Literal['channel last', 'channel first', 'none']]
    minibatch_input: list[bool]
    data_type: list[str]
    intensity_range: list[Literal['full', 'normalized_unsigned', 'normalized_signed']]
    device: list[str]


bunch_of_img_repr: dict[ImageRepr, ValuesOfImgRepr] = {
    "torch.tensor": {
        "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
        "channel_order": ['channel last', 'channel first', 'none'],
        "minibatch_input": [True, False],
        "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
                      'float32', 'float64', 'double',
                      'int8', 'int16', 'int32', 'int64'],
        "intensity_range": ['full', 'normalized_unsigned', 'normalized_signed'],
        "device": ['cpu', 'gpu']
    },
    "numpy.ndarray": {
        "color_channel": ['rgb', 'bgr', 'gray'],
        "channel_order": ['channel last', 'channel first', 'none'],
        "minibatch_input": [True, False],
        "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
                      'float32', 'float64', 'double',
                      'int8', 'int16', 'int32', 'int64'],
        "intensity_range": ['full', 'normalized_unsigned', 'normalized_signed'],
        "device": ['cpu']
    }
    # todo: add other data representations and values
}


def is_valid_value_of_img_repr(value: Metadata, valid_values: ValuesOfImgRepr):
    for attribute, values in valid_values.items():
        if value[attribute] not in values:
            return False
    return True


def find_closest_metadata(source_metadata, candidates):
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]

    targets = candidates
    targets = [
        candidate for candidate in targets
        if candidate["data_representation"] == source_metadata["data_representation"]
    ]
    if len(targets) == 0:
        targets = candidates

    color_matched_targets = [
        candidate for candidate in targets
        if candidate["color_channel"] == source_metadata["color_channel"]
    ]
    if len(color_matched_targets) == 0:
        if source_metadata["color_channel"] in ["rgb", "bgr"]:
            for metadata in targets:
                if metadata["color_channel"] in ["rgb", "bgr"]:
                    return metadata
    return targets[0]


def encode_metadata(metadata: dict) -> str:
    return '-'.join([str(metadata[key]) for key in Metadata.__annotations__.keys()])


def decode_metadata(metadata_str: str) -> dict:
    metadata_list = metadata_str.split('-')
    metadata = {k: v for k, v in zip(list(Metadata.__annotations__.keys()), metadata_list)}
    metadata['minibatch_input'] = True if metadata['minibatch_input'] == 'True' else False
    return metadata
