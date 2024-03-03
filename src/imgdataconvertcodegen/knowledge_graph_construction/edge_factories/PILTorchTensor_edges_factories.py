from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_only_this_key_differ


def pil_to_torch(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return (
            "from torchvision.transforms import ToTensor",
            "def convert(var):\n  return ToTensor()(var)",
        )
    return None


def torch_to_pil(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return (
            "from torchvision.transforms import ToPILImage",
            "def convert(var):\n  return ToPILImage()(var)",
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
        return tf.convert_to_tensor(np_array, dtype=tensorflow.uint8)""",
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


def torch_to_tf(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return (
            "import tensorflow as tf",
            """def convert(var):
                return tf.convert_to_tensor(var.numpy(), dtype=tf.as_dtype(var.dtype))""",
        )
    return None


def tf_to_torch(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_representation") == "tf.tensor"
        and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return (
            "import torch",
            """def convert(var):
                return torch.tensor(var_np = var.numpy(), dtype=var.dtype.as_numpy_dtype)""",
        )
    return None


def pil_rgba_to_rgb(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image"):
        return None
    if (
        source_metadata.get("color_channel") == "rgba"
        and target_metadata.get("color_channel") == "rgb"
    ) or (
        source_metadata.get("color_channel") == "graya"
        and target_metadata.get("color_channel") == "gray"
    ):
        if is_only_this_key_differ(source_metadata, target_metadata, "color_channel"):
            return (
                "",
                """def convert(var):
                    return var.convert("RGB") if var.mode == "RGBA" else var.convert("L")""",
            )
    return None


def pil_convert_dtype(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        target_dtype = target_metadata.get("data_type")

        return (
            "import numpy as np\nfrom PIL import Image",
            f"""def convert(var):
                    array = np.array(var)
                    if '{target_dtype}' == 'uint8':
                        converted = (array / 255.0 if array.dtype == np.float32 else array / 65535 if array.dtype == np.uint16 else array).astype(np.uint8)
                    elif '{target_dtype}' == 'uint16':
                        converted = (array * 65535 if array.dtype == np.uint8 else array).astype(np.uint16)
                    elif '{target_dtype}' == 'float32':
                        converted = (array / 255.0 if array.dtype == np.uint8 else array / 65535 if array.dtype == np.uint16 else array).astype(np.float32)
                    elif '{target_dtype}' == 'int32':
                        converted = array.astype(np.int32)

                    return Image.fromarray(converted) if '{target_dtype}' in ['uint8', 'uint16'] else converted""",
        )
    return None


def torch_gpu_cpu(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if (
        source_metadata.get("device") == "gpu"
        and target_metadata.get("device") == "cpu"
    ):
        if is_only_this_key_differ(source_metadata, target_metadata, "device"):
            return "import torch", "def convert(var):\n  return var.cpu()"

    if (
        source_metadata.get("device") == "cpu"
        and target_metadata.get("device") == "gpu"
    ):
        if is_only_this_key_differ(source_metadata, target_metadata, "device"):
            return "import torch", "def convert(var):\n  return var.cuda()"
    return None


def torch_channel_order_first_to_last(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "channel last"
    ):
        if source_metadata.get("minibatch_input"):
            return "import torch", "def convert(var):\n  return var.permute(0, 2, 3, 1)"
        return "import torch", "def convert(var):\n  return var.permute(1, 2, 0)"
    return None


def torch_minibatch_input_true_to_false(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return "import torch", "def convert(var):\n  return var.squeeze(0)"

    return None


def torch_convert_dtype(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
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
        target_dtype = dtype_mapping.get(target_dtype_str, None)
        if target_dtype is None:
            raise ValueError(f"Unsupported target dtype: {target_dtype_str}")

        return (
            "import torch",
            f"""def convert(var):
                    if '{target_dtype}' == 'float':
                        max_val = 1.0
                        if var.dtype in [uint8, double, int8, int16, int32, int64]:
                            max_val = iinfo(var.dtype).max   
                    return (var / max_val).to({target_dtype})
                    else:
                        return var.to({target_dtype})""",
        )
    return None


def tf_gpu_cpu(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if (
        source_metadata.get("device") == "gpu"
        and target_metadata.get("device") == "cpu"
    ):
        if is_only_this_key_differ(source_metadata, target_metadata, "device"):
            return "def convert(var):\n  return tensorflow.device('/cpu:0')(var)"
    if (
        source_metadata.get("device") == "cpu"
        and target_metadata.get("device") == "gpu"
    ):
        if is_only_this_key_differ(source_metadata, target_metadata, "device"):
            return (
                "import tensorflow as tf",
                """def convert(var):
                    if not tf.config.list_physical_devices('GPU'):
                        raise RuntimeError("GPU device not available.")
                    with tf.device('/device:GPU:0'):
                        return tf.identity(var)""",
            )
    return None


def tf_convert_minibatch(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None

    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.squeeze(var, 0)",
        )

    if not source_metadata.get("minibatch_input") and target_metadata.get(
        "minibatch_input"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.expand_dims(var, 0)" "",
        )

    return None


def tf_convert_dtype(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        target_dtype_str = target_metadata.get("data_type")
        dtype_mapping = {
            "uint8": "tf.uint8",
            "uint16": "tf.uint16",
            "uint32": "tf.uint32",
            "uint64": "tf.uint64",
            "float16": "tf.float16",
            "float32": "tf.float32",
            "float64": "tf.float64",
            "int8": "tf.int8",
            "int16": "tf.int16",
            "int32": "tf.int32",
            "int64": "tf.int64",
            "double": "tf.float64",
        }
        target_dtype = dtype_mapping.get(target_dtype_str, None)
        if target_dtype is None:
            raise ValueError(f"Unsupported target dtype: {target_dtype_str}")

        return (
            "import tensorflow as tf",
            f"""def convert(var):
                    if '{target_dtype}' == 'tf.float32':
                        if var.dtype.is_integer:
                            max_val = tf.cast(tf.reduce_max(var), tf.float32)
                            return tf.cast(var, tf.float32) / max_val
                        else:
                            return tf.cast(var, tf.float32)
                    else:
                    
                        return tf.cast(var, {target_dtype})""",
        )
    return None


pil_factories = [
    pil_to_torch,
    pil_to_tf,
    pil_rgba_to_rgb,
    pil_convert_dtype,
    # todo: add more factories
]