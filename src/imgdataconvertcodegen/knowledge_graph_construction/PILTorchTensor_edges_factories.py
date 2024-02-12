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
