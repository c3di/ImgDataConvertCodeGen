from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_differ_value_for_key


def is_attribute_value_valid_for_torch(metadata):
    allowed_values = {
        "color_channel": ["rgb", "gray"],
        "channel_order": ["channel first", "channel last", "none"],
        "minibatch_input": [True, False],
        "data_type": [
            "uint8",
            "float32",
            "float64",
            "double",
            "int8",
            "int16",
            "int32",
            "int64",
        ],
        "intensity_range": ["full", "0to1"],
        "device": ["cpu", "gpu"],
    }
    for key, values in allowed_values.items():
        if key in metadata and metadata[key] not in values:
            return False
    return True


def is_metadata_valid_for_torch(metadata):
    if (
        metadata["data_type"]
        in ["uint8", "float64", "double", "int8", "int16", "int32", "int64"]
        and metadata["intensity_range"] != "full"
    ):
        return False
    if metadata["color_channel"] == "rgb" and metadata["channel_order"] == "none":
        return False
    return True


def can_use_factories_in_cluster(source_metadata, target_metadata):
    return (
        are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor")
        and is_attribute_value_valid_for_torch(source_metadata)
        and is_attribute_value_valid_for_torch(target_metadata)
        and is_metadata_valid_for_torch(source_metadata)
        and is_metadata_valid_for_torch(target_metadata)
    )


def gpu_to_cpu(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("device") == "gpu"
        and target_metadata.get("device") == "cpu"
    ):
        return "", "def convert(var):\n  return var.cpu()"
    return None


def cpu_to_gpu(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("device") == "cpu"
        and target_metadata.get("device") == "gpu"
    ):
        return "", "def convert(var):\n  return var.cuda()"
    return None


def channel_none_to_channel_first(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "none"
        and target_metadata.get("channel_order") == "channel first"
    ):
        return "", "def convert(var):\n  return var.unsqueeze(0)"
    return None


def channel_none_to_channel_last(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "none"
        and target_metadata.get("channel_order") == "channel last"
    ):
        return "", "def convert(var):\n  return var.unsqueeze(-1)"
    return None


def channel_last_to_none(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel last"
        and target_metadata.get("channel_order") == "none"
    ):
        return "", "def convert(var):\n  return var.squeeze(-1)"
    return None


def channel_first_to_none(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "none"
    ):
        return "", "def convert(var):\n  return var.squeeze(0)"
    return None


def channel_last_to_channel_first(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel last"
        and target_metadata.get("channel_order") == "channel first"
    ):
        if source_metadata.get("minibatch_input"):
            return "", "def convert(var):\n  return var.permute(0, 3, 1, 2)"
        return "", "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def channel_first_to_channel_last(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "channel last"
    ):
        if source_metadata.get("minibatch_input"):
            return "", "def convert(var):\n  return var.permute(0, 2, 3, 1)"
        return "", "def convert(var):\n  return var.permute(1, 2, 0)"
    return None


def minibatch_true_to_false(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return "", "def convert(var):\n  return var.squeeze(0)"
    return None


def minibatch_false_to_true(source_metadata, target_metadata) -> conversion:
    if (not source_metadata.get("minibatch_input")) and target_metadata.get(
        "minibatch_input"
    ):
        return "", "def convert(var):\n  return var.unsqueeze(0)"
    return None


def convert_dtype_without_rescale(source_metadata, target_metadata) -> conversion:
    if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
        dtype_mapping = {
            "uint8": "torch.uint8",
            "double": "torch.double",
            "int8": "torch.int8",
            "int16": "torch.int16",
            "int32": "torch.int32",
            "int64": "torch.int64",
            "float64": "torch.double",
            "float32": "torch.float",
        }
        return (
            "import tensorflow as tf\n  from torchvision.transforms import functional as F",
            f"""def convert(var):
    dtype = getattr(tf, '{dtype_mapping[target_metadata["data_type"]]}')
    return F.convert_image_dtype(var, dtype)""",
        )
    return None


def uint8_full_range_to_normalize(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return "", "def convert(var):\n  return var / 255"
    return None


def uint8_normalized_to_full_range(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "0to1"
        and target_metadata.get("intensity_range") == "full"
    ):
        return "", "def convert(var):\n  return var * 255"
    return None


def float32_full_range_to_normalize(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_type") == "float32"
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return (
            "",
            "def convert(var):\n  return (var - var.min()) / (var.max() - var.min())",
        )
    return None


def channel_first_rgb_to_gray(source_metadata, target_metadata) -> conversion:
    # [N, 3, H, W] -> [N, 1, H, W]
    if (
        source_metadata.get("channel_order") == "channel first"
        and source_metadata.get("color_channel") == "rgb"
        and target_metadata.get("color_channel") == "gray"
        and source_metadata.get("minibatch_input")
    ):
        return (
            "from torchvision.transforms import functional as F",
            "def convert(var):\n  return F.rgb_to_grayscale(var)",
        )
    return None


def channel_first_gray_to_rgb(source_metadata, target_metadata) -> conversion:
    # [N, 1, H, W] -> [N, 3, H, W]
    if (
        source_metadata.get("channel_order") == "channel first"
        and source_metadata.get("color_channel") == "gray"
        and target_metadata.get("color_channel") == "rgb"
        and source_metadata.get("minibatch_input")
    ):
        return (
            "import torch",
            "def convert(var):\n  return torch.cat((var, var, var), 1)",
        )
    return None


factories_cluster_for_Pytorch = (
    can_use_factories_in_cluster,
    [
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
        convert_dtype_without_rescale,
        uint8_full_range_to_normalize,
        uint8_normalized_to_full_range,
        float32_full_range_to_normalize,
        channel_first_rgb_to_gray,
        channel_first_gray_to_rgb,
    ],
)
