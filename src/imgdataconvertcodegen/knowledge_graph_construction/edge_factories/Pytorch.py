from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_differ_value_for_key


def is_valid_metadata_for_torch(metadata):
    if (metadata['data_type'] in ['float64', 'double', 'int8', 'int16', 'int32', 'int64']
            and metadata['intensity_range'] != "full"):
        return False
    if metadata['color_channel'] == 'rgb' and metadata['channel_order'] == 'none':
        return False

    allowed_values = {
        "color_channel": ['rgb', 'gray'],
        "channel_order": ['channel first', 'channel last', 'none'],
        "minibatch_input": [True, False],
        "data_type": ['uint8',
                      'float32', 'float64', 'double',
                      'int8', 'int16', 'int32', 'int64'],
        "intensity_range": ['full', '0to1'],
        "device": ['cpu', 'gpu']
    }
    for key, values in allowed_values.items():
        if key in metadata and metadata[key] not in values:
            return False
    return True


def use_factories_in_cluster(source_metadata, target_metadata):
    return (are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor") and
            is_valid_metadata_for_torch(source_metadata) and
            is_valid_metadata_for_torch(target_metadata))


def gpu_to_cpu(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("device") == "gpu" and target_metadata.get("device") == "cpu":
        return "", "def convert(var):\n  return var.cpu()"
    return None


def cpu_to_gpu(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("device") == "cpu" and target_metadata.get("device") == "gpu":
        return "", "def convert(var):\n  return var.cuda()"
    return None


def channel_none_to_channel_first(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("channel_order") == "none" and target_metadata.get("channel_order") == "channel first":
        return "", "def convert(var):\n  return var.unsqueeze(0)"


def channel_none_to_channel_last(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("channel_order") == "none" and target_metadata.get("channel_order") == "channel last":
        return "", "def convert(var):\n  return var.unsqueeze(-1)"
    return None


def channel_last_to_none(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("channel_order") == "channel last" and target_metadata.get("channel_order") == "none":
        return "", "def convert(var):\n  return var.squeeze(-1)"
    return None


def channel_first_to_none(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("channel_order") == "channel first" and target_metadata.get("channel_order") == "none":
        return "", "def convert(var):\n  return var.squeeze(0)"
    return None


def channel_last_to_channel_first(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("channel_order") == "channel last" and target_metadata.get(
            "channel_order") == "channel first":
        if source_metadata.get("minibatch_input"):
            return "", "def convert(var):\n  return var.permute(0, 3, 1, 2)"
        return "", "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def channel_first_to_channel_last(source_metadata, target_metadata) -> conversion:
    if source_metadata.get('channel_order') == 'channel first' and target_metadata.get(
            'channel_order') == 'channel last':
        if source_metadata.get('minibatch_input'):
            return "", "def convert(var):\n  return var.permute(0, 2, 3, 1)"
        return "", "def convert(var):\n  return var.permute(1, 2, 0)"
    return None


def minibatch_true_to_false(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("minibatch_input") and not target_metadata.get("minibatch_input"):
        return "", "def convert(var):\n  return var.squeeze(0)"
    return None


def minibatch_false_to_true(source_metadata, target_metadata) -> conversion:
    if (not source_metadata.get('minibatch_input')) and target_metadata.get('minibatch_input'):
        return "", "def convert(var):\n  return var.unsqueeze(0)"


def convert_dtype(source_metadata, target_metadata) -> conversion:
    if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
        dtype_mapping = {
            "uint8": "uint8",
            "double": "double",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "float64": "double",
            "float32": "float",
        }
        return "", f'def convert(var):\n  return var.to("{dtype_mapping.get(target_metadata.get("data_type"))}")'
    return None


def uint8_data_range_to_normalize(source_metadata, target_metadata) -> conversion:
    if (source_metadata.get('data_type') == 'uint8' and
            source_metadata.get("intensity_range") == "full" and
            target_metadata.get("intensity_range") == "0to1"):
        return "", "def convert(var):\n  return var / 255"


def uint8_normalize_to_full_data_range(source_metadata, target_metadata) -> conversion:
    if (source_metadata.get('data_type') == 'uint8' and
            source_metadata.get("intensity_range") == "0to1" and
            target_metadata.get("intensity_range") == "full"):
        return "", "def convert(var):\n  return var * 255"
    return None


factories_cluster_for_Pytorch = (use_factories_in_cluster, [
    gpu_to_cpu,
    cpu_to_gpu,
    channel_none_to_channel_first,
    channel_none_to_channel_last,
    channel_first_to_none,
    channel_first_to_channel_last,
    channel_last_to_none,
    channel_last_to_channel_first,
    minibatch_true_to_false,
    minibatch_false_to_true,
    convert_dtype,
    uint8_data_range_to_normalize,
    uint8_normalize_to_full_data_range,
])
