from .type import conversion
from ...metadata_differ import are_both_same_data_repr, is_differ_value_for_key


def tf_gpu_cpu(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if (
            source_metadata.get("device") == "gpu"
            and target_metadata.get("device") == "cpu"
    ):
        if is_differ_value_for_key(source_metadata, target_metadata, "device"):
            return ("import tensorflow as tf",
                    "def convert(var):\n  return tf.device('/cpu:0')(var)")
    if (
            source_metadata.get("device") == "cpu"
            and target_metadata.get("device") == "gpu"
    ):
        if is_differ_value_for_key(source_metadata, target_metadata, "device"):
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
            "def convert(var):\n  return tf.expand_dims(var, 0)",
        )

    return None


def tf_convert_dtype(source_metadata, target_metadata) -> conversion:
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
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


tensorflow_factories = [
    tf_gpu_cpu,
    tf_convert_minibatch,
    tf_convert_dtype,
]
