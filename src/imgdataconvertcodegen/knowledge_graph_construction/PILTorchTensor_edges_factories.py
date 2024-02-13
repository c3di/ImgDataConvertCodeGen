from ..metadata_differ import are_both_same_data_repr, is_only_this_key_differ


def pil_to_torch(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return "def convert(var):\n  return torchvision.transforms.ToTensor()(var)"
    return None


def torch_to_pil(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return "def convert(var):\n  return torchvision.transforms.ToPILImage()(var)"
    return None


def pil_to_tf(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return """def convert(var):
    np_array = numpy.array(var)
    return tensorflow.convert_to_tensor(np_array, dtype=tensorflow.uint8)"""
    return None


def tf_to_pil(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "tf.tensor"
        and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return "def convert(var):\n  return PIL.Image.fromarray(var.numpy())"
    return None


def torch_to_tf(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return """def convert(var):
        return tensorflow.convert_to_tensor(var.numpy(), dtype=tensorflow.as_dtype(var.dtype))"""
    return None


def tf_to_torch(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "tf.tensor"
        and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return """def convert(var):
        return torch.tensor(var_np = var.numpy(), dtype=var.dtype.as_numpy_dtype)"""
    return None


def pil_rgba_to_rgb(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "color_channel"):
        if (
            source_metadata.get("color_channel") == "rgba"
            and target_metadata.get("color_channel") == "rgb"
        ) or (
            source_metadata.get("color_channel") == "graya"
            and target_metadata.get("color_channel") == "gray"
        ):
            return """def convert(var):
        return var.convert("RGB") if var.mode == "RGBA" else var.convert("L")"""
    return None


def pil_convert_dtype(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        target_dtype = target_metadata.get("data_type")

        return f"""def convert(var):
        array = numpy.array(var)
        if '{target_dtype}' == 'uint8':
            converted = (array / 255.0 if array.dtype == numpy.float32 else array / 65535 if array.dtype == numpy.uint16 else array).astype(numpy.uint8)
        elif '{target_dtype}' == 'uint16':
            converted = (array * 65535 if array.dtype == numpy.uint8 else array).astype(numpy.uint16)
        elif '{target_dtype}' == 'float32':
            converted = (array / 255.0 if array.dtype == numpy.uint8 else array / 65535 if array.dtype == numpy.uint16 else array).astype(numpy.float32)
        elif '{target_dtype}' == 'int32':
            converted = array.astype(numpy.int32)

        return PIL.Image.fromarray(converted) if '{target_dtype}' in ['uint8', 'uint16'] else converted"""
    return None


def torch_gpu_cpu(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "device"):
        if (
            source_metadata.get("device") == "gpu"
            and target_metadata.get("device") == "cpu"
        ):
            return "def convert(var):\n  return var.cpu()"
        if (
            source_metadata.get("device") == "cpu"
            and target_metadata.get("device") == "gpu"
        ):
            return "def convert(var):\n  return var.cuda()"
    return None


def torch_channel_order_first_to_last(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if (
        source_metadata.get("channel_order") == "channel first"
        and target_metadata.get("channel_order") == "channel last"
    ):
        if source_metadata.get("minibatch_input"):
            return "def convert(var):\n  return var.permute(0, 2, 3, 1)"
        return "def convert(var):\n  return var.permute(1, 2, 0)"
    return None


def torch_minibatch_input_true_to_false(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return "def convert(var):\n  return var.squeeze(0)"

    return None


def torch_convert_dtype(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "torch.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        target_dtype_str = target_metadata.get("data_type")
        dtype_mapping = {
            "uint8": "torch.uint8",
            "float": "torch.float",
            "double": "torch.double",
            "int8": "torch.int8",
            "int16": "torch.int16",
            "int32": "torch.int32",
            "int64": "torch.int64",
        }
        target_dtype = dtype_mapping.get(target_dtype_str, None)
        if target_dtype is None:
            raise ValueError(f"Unsupported target dtype: {target_dtype_str}")

        return f"""def convert(var):
    if '{target_dtype}' == 'torch.float':
        max_val = 1.0
        if var.dtype in [torch.uint8, torch.double, torch.int8, torch.int16, torch.int32, torch.int64]:
            max_val = torch.iinfo(var.dtype).max   
    return (var / max_val).to({target_dtype})
    else:
        return var.to({target_dtype})"""
    return None


def tf_gpu_cpu(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "device"):
        if (
            source_metadata.get("device") == "gpu"
            and target_metadata.get("device") == "cpu"
        ):
            return "def convert(var):\n  return tensorflow.device('/cpu:0')(var)"
        if (
            source_metadata.get("device") == "cpu"
            and target_metadata.get("device") == "gpu"
        ):
            return """def convert(var):
    if not tensorflow.config.list_physical_devices('GPU'):
        raise RuntimeError("GPU device not available.")
    with tensorflow.device('/device:GPU:0'):
        return tensorflow.identity(var)"""
    return None


def tf_convert_minibatch(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None

    if source_metadata.get("minibatch_input") and not target_metadata.get(
        "minibatch_input"
    ):
        return "def convert(var):\n  return tensorflow.squeeze(var, 0)"

    if not source_metadata.get("minibatch_input") and target_metadata.get(
        "minibatch_input"
    ):
        return "def convert(var):\n  return tensorflow.expand_dims(var, 0)"""

    return None


def tf_convert_dtype(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, "tf.tensor"):
        return None
    if is_only_this_key_differ(source_metadata, target_metadata, "data_type"):
        target_dtype_str = target_metadata.get("data_type")
        dtype_mapping = {
            "uint8": "tensorflow.uint8",
            "uint16": "tensorflow.uint16",
            "uint32": "tensorflow.uint32",
            "uint64": "tensorflow.uint64",
            "float16": "tensorflow.float16",
            "float32": "tensorflow.float32",
            "float64": "tensorflow.float64",
            "int8": "tensorflow.int8",
            "int16": "tensorflow.int16",
            "int32": "tensorflow.int32",
            "int64": "tensorflow.int64",
        }
        target_dtype = dtype_mapping.get(target_dtype_str, None)
        if target_dtype is None:
            raise ValueError(f"Unsupported target dtype: {target_dtype_str}")

        return f"""def convert(var):
    
    if '{target_dtype}' == 'tensorflow.float32':
        if var.dtype.is_integer:
            max_val = tensorflow.cast(tensorflow.reduce_max(var), tensorflow.float32)
            return tensorflow.cast(var, tensorflow.float32) / max_val
        else:
            return tensorflow.cast(var, tensorflow.float32)
    else:
    
        return tensorflow.cast(var, {target_dtype})"""
    return None
