from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_differ_value_for_key


def validate_torch(metadata):
    torch_constraints = {
        "color_channel": ['rgb', 'gray'],
        "channel_order": ['channel first', 'channel last', 'none'],
        "minibatch_input": [True, False],
        "data_type": ['uint8',
                      'float32', 'float64', 'double',
                      'int8', 'int16', 'int32', 'int64'],
        "intensity_range": ['full', 'normalized_unsigned'],
        "device": ['cpu', 'gpu']
    }
    if (metadata['data_type'] in ['uint8', 'float64', 'double', 'int8', 'int16', 'int32', 'int64']
            and metadata['intensity_range'] != "full"):
        return False
    for key, allowed_values in torch_constraints.items():
        if key in metadata and metadata[key] not in allowed_values:
            return False
    return True



def torch_gpu_cpu(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        if validate_torch(source_metadata) and validate_torch(target_metadata):
            if (
                    source_metadata.get("device") == "gpu"
                    and target_metadata.get("device") == "cpu"
            ):
                if is_differ_value_for_key(source_metadata, target_metadata, "device"):
                    return "import torch", "def convert(var):\n  return var.cpu()"

            if (
                    source_metadata.get("device") == "cpu"
                    and target_metadata.get("device") == "gpu"
            ):
                if is_differ_value_for_key(source_metadata, target_metadata, "device"):
                    return "import torch", "def convert(var):\n  return var.cuda()"
    return None


def torch_channel_order(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        if validate_torch(source_metadata) and validate_torch(target_metadata):
            if source_metadata.get("color_channel") == "gray":
                if target_metadata.get("channel_order") == "none":
                    if is_differ_value_for_key(source_metadata, target_metadata, "channel_order"):
                        if source_metadata.get("channel_order") == "channel first":
                            return "import torch", "def convert(var):\n  return var.squeeze(0)"
                        return "import torch", "def convert(var):\n  return var.squeeze(-1)"
                    if source_metadata.get("channel_order") == "none":
                        if is_differ_value_for_key(source_metadata, target_metadata, "channel_order"):
                            if target_metadata.get("channel_order") == "channel first":
                                return "import torch", "def convert(var):\n  return var.unsqueeze(0)"
                            return "import torch", "def convert(var):\n  return var.unsqueeze(-1)"
            if (
                    source_metadata.get("channel_order") == "channel first"
                    and target_metadata.get("channel_order") == "channel last"):
                if is_differ_value_for_key(source_metadata, target_metadata, "channel_order"):
                    if source_metadata.get("minibatch_input"):
                        return "import torch", "def convert(var):\n  return var.permute(0, 2, 3, 1)"
                    return "import torch", "def convert(var):\n  return var.permute(1, 2, 0)"
            if (
                    source_metadata.get('channel_order') == 'channel last' and
                    target_metadata.get('channel_order') == 'channel first'):
                if is_differ_value_for_key(source_metadata, target_metadata, "channel_order"):
                    if source_metadata.get("minibatch_input"):
                        return "import torch", "def convert(var):\n  return var.permute(0, 3, 1, 2)"
                    return "import torch", "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def torch_minibatch_input(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        if validate_torch(source_metadata) and validate_torch(target_metadata):
            if source_metadata.get("minibatch_input") and not target_metadata.get(
                    "minibatch_input"
            ):
                if is_differ_value_for_key(source_metadata, target_metadata, "minibatch_input"):
                    return "import torch", "def convert(var):\n  return var.squeeze(0)"
            if (not source_metadata.get('minibatch_input')) and target_metadata.get('minibatch_input'):
                if is_differ_value_for_key(source_metadata, target_metadata, "minibatch_input"):
                    return "import torch", "def convert(var):\n  return var.unsqueeze(0)"
    return None


def torch_convert_dtype_full(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        if validate_torch(source_metadata) and validate_torch(target_metadata):
            if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
                target_dtype_str = target_metadata.get("data_type")
                dtype_mapping = {
                    "uint8": "uint8",
                    "float32": "float",
                    "double": "double",
                    "int8": "int8",
                    "int16": "int16",
                    "int32": "int32",
                    "int64": "int64",
                    "float64": "double",
                }
                target_dtype = dtype_mapping.get(target_dtype_str)
                return (
                    "import torch",
                    f"def convert(var):\n  return var.to({target_dtype})",
                )
    return None


def torch_normalize(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        if validate_torch(source_metadata) and validate_torch(target_metadata):
            if (is_differ_value_for_key(source_metadata, target_metadata, "intensity_range")) and (
                    source_metadata['data_type'] == "float32" and source_metadata['intensity_range'] == 'full'):
                return (
                    "import torch",
                    "def convert(var):\n  return var - var.min() / (var.max() - var.min())"
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


pytorch_factories = [
    torch_gpu_cpu,
    torch_channel_order,
    torch_minibatch_input,
    torch_convert_dtype_full,
    torch_normalize,
    torch_to_pil,
    torch_to_tf,
]
