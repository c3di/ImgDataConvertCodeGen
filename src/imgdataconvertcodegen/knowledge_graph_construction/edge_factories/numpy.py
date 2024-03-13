from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_same_metadata


def is_metadata_valid_for_numpy(metadata):
    if (
        metadata["data_type"] in ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        and metadata["intensity_range"] != "full"
    ):
        return False
    if metadata["color_channel"] == "rgb" and metadata["channel_order"] == "none":
        return False
    return True


def is_attribute_value_valid_for_numpy(metadata):
    allowed_values = {
        "color_channel": ["rgb", "gray"],
        "channel_order": ["channel first", "channel last", "none"],
        "minibatch_input": [True, False],
        "data_type": [
            "uint8",
            "uint16",
            "uint32",
            "float32",
            "float64",
            "double",
            "int8",
            "int16",
            "int32",
        ],
        "intensity_range": ["full", "0to1", "-1to1"],
        "device": ["cpu"],
    }
    for key, values in allowed_values.items():
        if key in metadata and metadata[key] not in values:
            return False
    return True


def can_use_factories_in_cluster(source_metadata, target_metadata):
    return (
        are_both_same_data_repr(source_metadata, target_metadata, "numpy.ndarray")
        and is_attribute_value_valid_for_numpy(source_metadata)
        and is_attribute_value_valid_for_numpy(target_metadata)
        and is_metadata_valid_for_numpy(source_metadata)
        and is_metadata_valid_for_numpy(target_metadata)
    )


def channel_first_between_bgr_rgb(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel first":
        return None
    if (
        source_metadata["color_channel"] == "bgr"
        and target_metadata["color_channel"] == "rgb"
    ) or (
        source_metadata["color_channel"] == "rgb"
        and target_metadata["color_channel"] == "bgr"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, C, H, W]
            return (
                "import numpy as np",
                "def convert(var):\n  return var[:, ::-1, :, :]",
            )
        # [C, H, W]
        return "import numpy as np", "def convert(var):\n  return var[::-1, :, :]"
    return None


def channel_first_rgb_to_gray(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel first":
        return None
    if (
        source_metadata["color_channel"] == "rgb"
        and target_metadata["color_channel"] == "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, 3, H, W] -> [N, 1, H, W]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.sum(var * np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1), axis=1, keepdims=True)",
            )
        # [3, H, W] -> [1, H, W]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.sum(var * np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1), axis=0)",
        )
    return None


def channel_first_bgr_to_gray(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel first":
        return None
    if (
        source_metadata["color_channel"] == "bgr"
        and target_metadata["color_channel"] == "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, 3, H, W] -> [N, 1, H, W]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.sum(var * np.array([0.114, 0.587, 0.299]).reshape(1, 3, 1, 1), axis=1, keepdims=True)",
            )
        # [3, H, W] -> [1, H, W]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.sum(var * np.array([0.114, 0.587, 0.299]).reshape(3, 1, 1), axis=0)",
        )
    return None


def channel_first_gray_to_rgb_or_bgr(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel first":
        return None
    if (
        source_metadata["color_channel"] == "gray"
        and target_metadata["color_channel"] != "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, 1, H, W] -> [N, 3, H, W]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.repeat(var, 3, axis=1)",
            )
        # [1, H, W] -> [3, H, W]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.repeat(var, 3, axis=0)",
        )
    return None


def channel_last_between_bgr_rgb(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel last":
        return None
    if (
        source_metadata["color_channel"] == "bgr"
        and target_metadata["color_channel"] == "rgb"
    ) or (
        source_metadata["color_channel"] == "rgb"
        and target_metadata["color_channel"] == "bgr"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, H, W, C]
            return (
                "import numpy as np",
                "def convert(var):\n  return var[:, :, :, ::-1]",
            )
        # [H, W, C]
        return "import numpy as np", "def convert(var):\n  return var[:, :, ::-1]"


def channel_last_rgb_to_gray(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel last":
        return None
    if (
        source_metadata["color_channel"] == "rgb"
        and target_metadata["color_channel"] == "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, H, W, 3] -> [N, H, W, 1]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.dot(var[..., :3], [0.299, 0.587, 0.114]))",
            )
        # [H, W, 3] -> [H, W, 1]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.dot(var[:, :3], [0.299, 0.587, 0.114])",
        )
    return None


def channel_last_bgr_to_gray(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel last":
        return None
    if (
        source_metadata["color_channel"] == "bgr"
        and target_metadata["color_channel"] == "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, H, W, 3] -> [N, H, W, 1]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.dot(var[..., :3], [0.114, 0.587, 0.299])",
            )
        # [H, W, 3] -> [H, W, 1]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.dot(var[:, :3], [0.114, 0.587, 0.299])",
        )
    return None


def channel_last_gray_to_rgb_or_gbr(source_metadata, target_metadata) -> conversion:
    if source_metadata["channel_order"] != "channel last":
        return None
    if (
        source_metadata["color_channel"] == "gray"
        and target_metadata["color_channel"] != "gray"
    ):
        if source_metadata["mini_batch_input"]:
            # [N, H, W, 1] -> [N, H, W, 3]
            return (
                "import numpy as np",
                "def convert(var):\n  return np.repeat(var, 3, axis=-1)",
            )
        # [H, W, 1] -> [H, W, 3]
        return (
            "import numpy as np",
            "def convert(var):\n  return np.repeat(var, 3, axis=2)",
        )
    return None


def convert_dtype_without_rescale(source_metadata, target_metadata) -> conversion:
    if is_same_metadata(source_metadata, target_metadata, "data_type"):
        dtype_mapping = {
            "uint8": "np.uint8",
            "uint16": "np.uint16",
            "uint32": "np.uint32",
            "int8": "np.int8",
            "int16": "np.int16",
            "int32": "np.int32",
            "float32": "np.float32",
            "float64": "np.float64",
            "double": "np.float64",
        }
        return (
            "import numpy as np",
            f"""def convert(var):
    return var.astype({dtype_mapping[target_metadata["data_type"]]})""",
        )
    return None


def uint8_full_range_to_normalize_unsigned(
    source_metadata, target_metadata
) -> conversion:
    # only unsigned as it needs to be conversed to float to be normalized to signed
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return "", "def convert(var):\n  return var / 255"
    return None


def float_full_range_to_normalize_unsigned(
    source_metadata, target_metadata
) -> conversion:
    if (
        source_metadata.get("data_type") in ["float32", "float64", "double"]
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return (
            "import numpy as np",
            """def convert(var):
min_val = np.min(var)
max_val = np.max(var)
return (var - min_val) / (max_val - min_val)""",
        )
    return None


def float_full_range_to_normalize_signed(
    source_metadata, target_metadata
) -> conversion:
    if (
        source_metadata.get("data_type") in ["float32", "float64", "double"]
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "-1to1"
    ):
        return (
            "import numpy as np",
            """def convert(var):
min_val = np.min(var)
max_val = np.max(var)
return 2 * (var - min_val) / (max_val - min_val) - 1""",
        )
    return None


def float_normalized_unsigned_to_normalized_signed(
    source_metadata, target_metadata
) -> conversion:
    if (
        source_metadata.get("data_type") in ["float32", "float64", "double"]
        and source_metadata.get("intensity_range") == "0to1"
        and target_metadata.get("intensity_range") == "-1to1"
    ):
        return ("", "def convert(var):\n  return 2 * var - 1")
    return None


def float_normalized_signed_to_normalized_unsigned(
    source_metadata, target_metadata
) -> conversion:
    if (
        source_metadata.get("data_type") in ["float32", "float64", "double"]
        and source_metadata.get("intensity_range") == "-1to1"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return ("", "def convert(var):\n  return (var + 1) / 2")
    return None


def channel_last_to_channel_first(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'channel last' or target_metadata['channel_order'] == 'channel first':
        if source_metadata['mini_batch_input']:
            # [N, H, W, C] -> [N, C, H, W]
            return "", "def convert(var):\n  return var.permute(0, 3, 1, 2)"
        # [H, W, C] -> [C, H, W]
        return "", "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def channel_last_to_channel_none(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'channel last' or target_metadata['channel_order'] == 'none':
        # [N, H, W, 1] -> [N, H, W] or [H, W, 1] -> [H, W]
        return "", "def convert(var):\n  return var.squeeze(-1)"
    return None


def channel_first_to_channel_last(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'channel first' or target_metadata['channel_order'] == 'channel last':
        if source_metadata['mini_batch_input']:
            # [N, C, H, W] -> [N, H, W, C]
            return "", "def convert(var):\n  return var.permute(0, 2, 3, 1)"
        # [C, H, W] -> [H, W, C]
        return "", "def convert(var):\n  return var.permute(1, 2, 0)"
    return None


def channel_first_to_channel_none(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'channel first' or target_metadata['channel_order'] == 'none':
        if source_metadata['mini_batch_input']:
            # [N, 1, H, W] -> [N, H, W]
            return "", "def convert(var):\n  return var.squeeze(1)"
        # [1, H, W] -> [H, W]
        return "", "def convert(var):\n  return var.squeeze(0)"
    return None


def channel_none_to_channel_first(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'none' or target_metadata['channel_order'] == 'channel first':
        if source_metadata['mini_batch_input']:
            # [N, H, W] -> [N, 1, H, W]
            return "", "def convert(var):\n  return var.unsqueeze(1)"
        # [H, W] -> [1, H, W]
        return "", "def convert(var):\n  return var.unsqueeze(0)"
    return None


def channel_none_to_channel_last(source_metadata, target_metadata) -> conversion:
    if source_metadata['channel_order'] == 'none' or target_metadata['channel_order'] == 'channel last':
        # [N, H, W] -> [N, H, W, 1] or [H, W] -> [H, W, 1]
        return "", "def convert(var):\n  return var.unsqueeze(-1)"
    return None


def minibatch_true_to_false(source_metadata, target_metadata) -> conversion:
    if source_metadata['minibatch_input'] and not target_metadata['minibatch_input']:
        return "", "def convert(var):\n  return var[0]"
    return None


def minibatch_false_to_true(source_metadata, target_metadata) -> conversion:
    if not source_metadata['minibatch_input'] and target_metadata['minibatch_input']:
        return "", "def convert(var):\n  return var.unsqueeze(0)"
    return None


def uint8_normalized_unsigned_to_full_range(
    source_metadata, target_metadata
) -> conversion:
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "0to1"
        and target_metadata.get("intensity_range") == "full"
    ):
        return "", "def convert(var):\n  return var * 255"
    return None
#TODO: add more dtyps normalized to full range

factories_cluster_for_numpy = (
    can_use_factories_in_cluster,
    [
        channel_first_between_bgr_rgb,
        channel_first_rgb_to_gray,
        channel_first_bgr_to_gray,
        channel_first_gray_to_rgb_or_bgr,
        channel_last_between_bgr_rgb,
        channel_last_rgb_to_gray,
        channel_last_bgr_to_gray,
        channel_last_gray_to_rgb_or_gbr,
        channel_last_to_channel_first,
        channel_last_to_channel_none,
        channel_first_to_channel_last,
        channel_first_to_channel_none,
        channel_none_to_channel_first,
        channel_none_to_channel_last,
        minibatch_true_to_false,
        minibatch_false_to_true,
        convert_dtype_without_rescale,
        uint8_full_range_to_normalize_unsigned,
        float_full_range_to_normalize_unsigned,
        float_full_range_to_normalize_signed,
        float_normalized_unsigned_to_normalized_signed,
        float_normalized_signed_to_normalized_unsigned,
        uint8_normalized_unsigned_to_full_range,
    ],
)
