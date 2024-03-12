from .type import conversion
from ...metadata_differ import is_differ_value_for_key


def pil_to_torch(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "PIL.Image"
            and target_metadata.get("data_representation") == "torch.tensor"
    ):
        if is_differ_value_for_key(source_metadata, target_metadata, "data_representation"):
            common_constraints = {
                "color_channel": ['rgb', 'gray'],
                "channel_order": ['channel last', 'channel first', 'none'],
                "minibatch_input": [False],
                "data_type": ['uint8'],
                "device": ['cpu']
            }

            for key, allowed_values in common_constraints.items():
                if not source_metadata.get(key) in allowed_values:
                    return None

            return (
                "from torchvision.transforms import ToTensor",
                "def convert(var):\n  return ToTensor()(var)",
            )
        return None


def pil_to_tf(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "PIL.Image"
            and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return (
            "import tensorflow as tf\nimport numpy as np",
            """def convert(var):
        np_array = np.array(var)
        return tf.convert_to_tensor(np_array, dtype=tf.uint8)""",
        )
    return None


def torch_to_pil(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "torch.tensor"
            and target_metadata.get("data_representation") == "PIL.Image"
    ):
        if is_differ_value_for_key(source_metadata, target_metadata, "data_representation"):
            common_constraints = {
                "color_channel": ['rgb', 'gray'],
                "channel_order": ['channel last', 'channel first', 'none'],
                "minibatch_input": [False],
                "data_type": ['uint8'],
                "device": ['cpu']
            }

            for key, allowed_values in common_constraints.items():
                if not source_metadata.get(key) in allowed_values:
                    return None
        return (
            "from torchvision.transforms import ToPILImage",
            "def convert(var):\n  return ToPILImage()(var)",
        )
    return None


def torch_to_tf(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "torch.tensor"
            and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.convert_to_tensor(var.numpy(), dtype=tf.as_dtype(var.dtype))",
        )
    return None


def tf_to_pil(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "tf.tensor"
            and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return (
            "from PIL import Image",
            "def convert(var):\n  return Image.fromarray(var.numpy())",
        )
    return None


def tf_to_torch(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "tf.tensor"
            and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return (
            "import torch",
            "def convert(var):\n  return torch.tensor(var_np = var.numpy(), dtype=var.dtype.as_numpy_dtype)",
        )
    return None


inter_libs_factories = [
    pil_to_torch,
    pil_to_tf,
    torch_to_pil,
    torch_to_tf,
    tf_to_pil,
    tf_to_torch
]
