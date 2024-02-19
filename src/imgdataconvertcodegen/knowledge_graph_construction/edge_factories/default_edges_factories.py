from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_only_this_key_differ


# NOTE: the source and target metadata are only different in one attribute


# float64 and double same for "full" intensity_range:
# numpy.ndarray:
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64 :
# numpy.float64 alias of double
# ---
# torch.tensor:
# https://pytorch.org/docs/stable/tensors.html :
# 64-bit floating point: torch.float64 or torch.double
# ---
# tf.tensor:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/dtypes.py#L387C1-L387C17
# (16.02.2024) line 387: double = float64
# ===
# but intensity_range can be also -1 to 1 or 0 to 1, because of scikit-image:
# https://scikit-image.org/docs/stable/user_guide/data_types.html
# For scikit-image, it is unconventional to call it double, but we leave that possibility.
def between_float64_double(source_metadata, target_metadata) -> conversion:
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        if ((source_metadata.get('data_type') == 'float64' and
                target_metadata.get('data_type') == 'double') or
            (source_metadata.get('data_type') == 'double' and
                target_metadata.get('data_type') == 'float64')):
            return "", "def convert(var):\n  return var"
    return None


def numpy_between_rgb_bgr(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, 'numpy.ndarray'):
        return None
    if ((source_metadata.get('color_channel') == 'bgr' and
        target_metadata.get('color_channel') == 'rgb') or
        (source_metadata.get('color_channel') == 'rgb' and
         target_metadata.get('color_channel') == 'bgr')):
        return "", "def convert(var):\n  return var[:, :, ::-1]"
    return None


def numpy_to_torch(source_metadata, target_metadata) -> conversion:
    if (source_metadata.get('data_representation') == 'numpy.ndarray' and
            target_metadata.get('data_representation') == 'torch.tensor'):
        return "import torch", "def convert(var):\n  return torch.from_numpy(var)"
    return None


def torch_channel_order_last_to_first(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, 'torch.tensor'):
        return None
    if (source_metadata.get('channel_order') == 'channel last' and
            target_metadata.get('channel_order') == 'channel first'):
        if source_metadata.get('minibatch_input'):
            return "", "def convert(var):\n  return var.permute(0, 3, 1, 2)"
        return "", "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def torch_minibatch_input_false_to_true(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, 'torch.tensor'):
        return None
    if (not source_metadata.get('minibatch_input')) and target_metadata.get('minibatch_input'):
        return "import torch", "def convert(var):\n  return torch.unsqueeze(var, 0)"

    return None


# Todo... add more factories from the table
default_factories = [
    between_float64_double,
    numpy_between_rgb_bgr,
    numpy_to_torch,
    torch_channel_order_last_to_first,
    torch_minibatch_input_false_to_true]
