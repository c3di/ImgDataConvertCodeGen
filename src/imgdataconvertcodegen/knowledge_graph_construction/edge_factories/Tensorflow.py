from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_differ_value_for_key


def is_attribute_value_valid_for_tensorflow(metadata):
    # reference: https://www.tensorflow.org/api_docs/python/tf/dtypes
    allowed_values = {
        "color_channel": ["rgb", "gray"],
        "channel_order": ["channel first", "channel last", "none"],
        "minibatch_input": [True, False],
        "data_type": [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
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


def is_metadata_valid_for_tensorflow(metadata):
    if (
        metadata["data_type"]
        in [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float64",
            "double",
            "int8",
            "int16",
            "int32",
            "int64",
        ]
        and metadata["intensity_range"] != "full"
    ):
        return False

    if metadata["color_channel"] == "rgb" and metadata["channel_order"] == "none":
        return False

    return True


def can_use_factories_in_cluster(source_metadata, target_metadata):
    return (
            are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor")
            and is_attribute_value_valid_for_tensorflow(source_metadata)
            and is_attribute_value_valid_for_tensorflow(target_metadata)
            and is_metadata_valid_for_tensorflow(source_metadata)
            and is_metadata_valid_for_tensorflow(target_metadata)
    )


def gpu_to_cpu(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("device") == "gpu"
        and target_metadata.get("device") == "cpu"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.device('/cpu:0')(var)",
        )
    return None


def cpu_to_gpu(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("device") == "cpu"
        and target_metadata.get("device") == "gpu"
    ):
        return (
                "import tensorflow as tf",
                """def convert(var):
    with tf.device('/device:GPU:0'):
        return tf.identity(var)""",
        )
    return None


def minibatch_true_to_false(source_metadata, target_metadata) -> conversion:
    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.squeeze(var, 0)",
        )
    return None


def minibatch_false_to_true(source_metadata, target_metadata) -> conversion:
    if (not source_metadata.get("minibatch_input")) and target_metadata.get(
        "minibatch_input"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.expand_dims(var, 0)",
        )
    return None


def channel_none_to_channel_first(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "none"
        and target_metadata.get("channel_order") == "channel first"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.expand_dims(var, 0)",
        )
    return None


def channel_none_to_channel_last(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "none"
        and target_metadata.get("channel_order") == "channel last"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.expand_dims(var, -1)",
        )
    return None


def channel_last_to_none(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel last"
        and target_metadata.get("channel_order") == "none"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.squeeze(var, -1)",
        )
    return None


def channel_first_to_none(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "none"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.squeeze(var, 0)",
        )
    return None


def channel_last_to_channel_first(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel last"
        and target_metadata.get("channel_order") == "channel first"
    ):
        if source_metadata.get("minibatch_input"):
            return (
                "import tensorflow as tf",
                "def convert(var):\n  return tf.transpose(var, [0, 3, 1, 2])",
            )
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.transpose(var, [2, 0, 1])",
        )
    return None


def channel_first_to_channel_last(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "channel last"
    ):
        if source_metadata.get("minibatch_input"):
            return (
                "import tensorflow as tf",
                "def convert(var):\n  return tf.transpose(var, [0, 2, 3, 1])",
            )
        return (
            "import tensorflow as tf",
            "def convert(var):\n   return tf.transpose(var, [1, 2, 0])",
        )
    return None


def convert_dtype_without_rescale(source_metadata, target_metadata) -> conversion:
    if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
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
        target_dtype = dtype_mapping.get(target_metadata.get("data_type"))
        return (
            "import tensorflow as tf",
            f"""def convert(var):
    dtype = getattr(tf, '{target_dtype}')
    return tf.cast(var, dtype)""",
        )
    return None


def uint8_data_range_to_normalize(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "full"
        and target_metadata.get("intensity_range") == "0to1"
    ):
        return "", "def convert(var):\n  return var / 255"
    return None


def uint8_normalize_to_full_data_range(source_metadata, target_metadata) -> conversion:
    if (
        source_metadata.get("data_type") == "uint8"
        and source_metadata.get("intensity_range") == "0to1"
        and target_metadata.get("intensity_range") == "full"
    ):
        return "", "def convert(var):\n  return var * 255"
    return None


def float32_full_to_float32_0to1(source_metadata, target_metadata) -> conversion:
    # Todo
    pass


def float32_0to1_to_float32_full(source_metadata, target_metadata) -> conversion:
    pass


def channel_last_rgb_to_gray(source_metadata, target_metadata) -> conversion:
    # Outputs a tensor of the same `DType`. The operation supports data types (for `image` and `dtype`) of `uint8`,
    # `uint16`, `uint32`, `uint64`, `int8`,`int16`, `int32`, `int64`, `float16`, `float32`, `float64`.
    # Images that are represented using floating point values are expected to have values in the range [0,1).
    # Image data stored in integer data types are expected to have values in the range `[0,MAX]`,
    # where `MAX` is the largest positive representable number for the data type.
    # This op converts between data types, scaling the values appropriately before casting.
    # reference: https://github.com/tensorflow/tensorflow/blob/0f7eb923815f4fd82321dda01bab45b493227d58/tensorflow/python/ops/image_ops_impl.py#L2381-L2393
    is_allowed_dtype_intensity_range = ((source_metadata.get("data_type") == "float32" and
                                         source_metadata.get("intensity_range") == "0to1")
                                        or source_metadata.get("data_type") != "float32")
    # [N, H, W, 3] -> [N, H, W, 1]
    if (
            source_metadata.get("channel_order") == "channel last" and
            source_metadata.get("color_channel") == "rgb" and
            target_metadata.get("color_channel") == "gray" and
            source_metadata.get("minibatch_input") and
            is_allowed_dtype_intensity_range
    ):
        return "import tensorflow as tf", "def convert(var):\n  return tf.image.rgb_to_grayscale(var)"
    return None


def channel_last_gray_to_rgb(source_metadata, target_metadata) -> conversion:
    # [N, H, W, 1] -> [N, H, W, 3]
    # Outputs a tensor of the same `DType`
    if (
            source_metadata.get("channel_order") == "channel last" and
            source_metadata.get("color_channel") == "gray" and
            target_metadata.get("color_channel") == "rgb" and
            source_metadata.get("minibatch_input")):
        return "import tensorflow as tf", "def convert(var):\n  return tf.image.grayscale_to_rgb(var)"
    return None


factories_cluster_for_tensorflow = (
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
        uint8_data_range_to_normalize,
        uint8_normalize_to_full_data_range,
        channel_last_rgb_to_gray,
        channel_last_gray_to_rgb
    ],
)
