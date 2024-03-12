from .type import conversion
from ...metadata_differ import is_same_metadata


def numpy_to_torch(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["data_type"] in ["uint8", "int8", "int16", "int32"]
        and target_metadata["intensity_range"] != "full"
    ) or (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] == "none"
    ):
        return None

    if (
        source_metadata.get("data_representation") == "numpy.ndarray"
        and target_metadata.get("data_representation") == "torch.tensor"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
    ):
        common_values_numpy_torch = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "channel first", "none"],
            "minibatch_input": [False],
            "data_type": ["uint8", "float32", "int8", "int16", "int32"],
            "intensity_range": ["full", "0to1"],
            "device": ["cpu"],
        }
        for key, allowed_values in common_values_numpy_torch.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "import torch",
            "def convert(var):\n  return torch.from_numpy(var)",
        )
    return None


def torch_to_numpy(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["data_type"] in ["uint8", "int8", "int16", "int32"]
        and target_metadata["intensity_range"] != "full"
    ) or (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] == "none"
    ):
        return None
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "numpy.ndarray"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
        and (
            source_metadata["data_type"] in ["uint8", "int8", "int16", "int32"]
            and target_metadata["intensity_range"] == "full"
        )
    ):
        common_values_numpy_torch = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "channel first", "none"],
            "minibatch_input": [False],
            "data_type": ["uint8", "float32", "int8", "int16", "int32"],
            "intensity_range": ["full", "0to1"],
            "device": ["cpu"],
        }
        for key, allowed_values in common_values_numpy_torch.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "import torch",
            "def convert(var):\n  return var.numpy(force=True)",
        )
    return None


def numpy_to_pil(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] != "channel last"
    ):
        return None
    if (
        source_metadata.get("data_representation") == "numpy.ndarray"
        and target_metadata.get("data_representation") == "PIL.Image"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
    ):
        common_constraints_numpy_pil = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "none"],
            "minibatch_input": [False],
            "data_type": ["uint8"],
            "intensity_range": ["full"],
            "device": ["cpu"],
        }

        for key, allowed_values in common_constraints_numpy_pil.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "from PIL import Image",
            "def convert(var):\n  return Image.fromarray(var)",
        )
    return None


def pil_to_numpy(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] != "channel last"
    ):
        return None
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "numpy.ndarray"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
    ):
        common_constraints_numpy_pil = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "none"],
            "minibatch_input": [False],
            "data_type": ["uint8"],
            "intensity_range": ["full"],
            "device": ["cpu"],
        }

        for key, allowed_values in common_constraints_numpy_pil.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "import numpy as np",
            "def convert(var):\n  return np.array(var)",
        )
    return None


def tensorflow_to_numpy(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] != "channel last"
    ):
        return None
    if (
        source_metadata["data_type"] == "float32"
        and source_metadata["intensity_range"] != "0to1"
    ) or (
        source_metadata["data_type"]
        in ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        and source_metadata["intensity_range"] != "full"
    ):
        return None
    if (
        source_metadata.get("data_representation") == "tf.tensor"
        and target_metadata.get("data_representation") == "numpy.ndarray"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
    ):
        common_values_numpy_tensorflow = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "none"],
            "minibatch_input": [False],
            "data_type": [
                "uint8",
                "uint16",
                "uint32",
                "float32",
                "int8",
                "int16",
                "int32",
            ],
            "intensity_range": ["full", "0to1"],
            "device": ["cpu"],
        }
        for key, allowed_values in common_values_numpy_tensorflow.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.make_ndarray(var)",
        )
    return None


def numpy_to_tensorflow(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata["color_channel"] == "rgb"
        and source_metadata["channel_order"] != "channel last"
    ):
        return None
    if (
        source_metadata["data_type"] == "float32"
        and source_metadata["intensity_range"] != "0to1"
    ) or (
        source_metadata["data_type"]
        in ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        and source_metadata["intensity_range"] != "full"
    ):
        return None
    if (
        source_metadata.get("data_representation") == "numpy.ndarray"
        and target_metadata.get("data_representation") == "tf.tensor"
        and is_same_metadata(source_metadata, target_metadata, "data_representation")
    ):
        common_values_numpy_tensorflow = {
            "color_channel": ["rgb", "gray"],
            "channel_order": ["channel last", "none"],
            "minibatch_input": [False],
            "data_type": [
                "uint8",
                "uint16",
                "uint32",
                "float32",
                "int8",
                "int16",
                "int32",
            ],
            "intensity_range": ["full", "0to1"],
            "device": ["cpu"],
        }
        for key, allowed_values in common_values_numpy_tensorflow.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "import tensorflow as tf",
            f"def convert(var):\n  return tf.convert_to_tensor(var, dtype ={source_metadata['data_type']})",
        )
    return None


inter_libs_factories = [
    numpy_to_pil,
    pil_to_numpy,
    numpy_to_torch,
    torch_to_numpy,
    numpy_to_tensorflow,
    tensorflow_to_numpy,
]
