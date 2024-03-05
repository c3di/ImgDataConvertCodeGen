from ..metadata_differ import is_same_metadata

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


def is_valid_metadata(metadata: dict):
    if metadata['color_channel'] == 'rgba' or metadata['color_channel'] == 'graya':
        return metadata['data_representation'] == 'PIL.Image'
    if metadata['data_representation'] == 'torch.tensor':
        for type in ['uint8', 'float32', 'int8', 'int16', 'int32', 'float64', 'double', 'int64']:
            if metadata['data_type'] == type:
                if metadata['data_type'] == 'float32' and metadata['intensity_range'] != 'normalized_unsigned':
                    return False
                return True
        return False
    return True


def is_valid_metadata_pair(source_metadata: dict, target_metadata: dict):
    return not is_same_metadata(source_metadata, target_metadata)


def check_metadata_value_valid(metadata: dict):
    for key, value_list in metadata_values.items():
        if not metadata.get(key) in value_list:
            raise ValueError(f'Invalid metadata: {metadata} at key: {key}. Expected value list: {value_list}')


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
    return '-'.join([str(v) for v in metadata.values()])


def decode_metadata(metadata_str: str) -> dict:
    metadata_list = metadata_str.split('-')
    metadata = {k: v for k, v in zip(metadata_values.keys(), metadata_list)}
    metadata['minibatch_input'] = True if metadata['minibatch_input'] == 'True' else False
    return metadata
