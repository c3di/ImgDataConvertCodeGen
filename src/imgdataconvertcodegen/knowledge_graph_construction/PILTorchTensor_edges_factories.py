from ..metadata_differ import are_both_same_data_repr, are_same_data_type


def pil_to_torch(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return "def convert(var):\n  return ToTensor()(var)"
    return None


def torch_to_pil(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "torch.tensor"
        and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return "def convert(var):\n  return ToPILImage()(var)"
    return None


def pil_to_tf(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "PIL.Image"
        and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return """def convert(var):
    import tensorflow as tf
    from PIL import ImageMode

    mode_to_dtype = {
        'L': numpy.uint8,    # 8-bit pixels, grayscale
        'P': numpy.uint8,    # 8-bit pixels, mapped to any other mode using a color palette
        'RGB': numpy.uint8,  # 3x8-bit pixels, true color
        'RGBA': numpy.uint8, # 4x8-bit pixels, true color with transparency mask
        'I': numpy.int32,    # 32-bit signed integer pixels
        'F': numpy.float32,  # 32-bit floating point pixels
        'I;16': numpy.uint16, # 16-bit pixels
        'I;16L': numpy.uint16, # 16-bit pixels, little-endian byte order
    }

    dtype = mode_to_dtype.get(var.mode, numpy.uint8) # Default to uint8 if mode not found
    np_array = numpy.array(var, dtype=dtype)
    tf_dtype = tf.as_dtype(np_array.dtype).as_numpy_dtype

    return tf.convert_to_tensor(np_array, dtype=tf_dtype)"""
    return None


def tf_to_pil(source_metadata, target_metadata):
    if (
        source_metadata.get("data_representation") == "tf.tensor"
        and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return """def convert(var):
    var_np = var.numpy() 
    if var_numpy.dtype == numpy.uint8:
        return Image.fromarray(var_np)
    elif var_numpy.dtype == numpy.uint16:
        return Image.fromarray(var_np, mode='I;16')
    elif var_numpy.dtype in [numpy.float32, numpy.float64]:
        return Image.fromarray(var_np, mode='F')
    elif var_numpy.dtype == numpy.int32:
        return Image.fromarray(var_np, mode='I')
    else:
        raise ValueError(f"Unsupported tensor datatype: {var_numpy.dtype}")"""

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

    if not are_same_data_type(source_metadata, target_metadata):
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

        return Image.fromarray(converted) if '{target_dtype}' in ['uint8', 'uint16'] else converted"""
