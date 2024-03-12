from .type import conversion
from ...metadata_differ import is_differ_value_for_key, are_both_same_data_repr


def validate_pil(metadata):
    pil_constraints = {
        "color_channel": ['rgb', 'gray', 'rgba', 'graya'],
        "channel_order": ['channel last', 'none'],
        "minibatch_input": [False],
        "data_type": ['uint8'],
        "intensity_range": ['full'],
        "device": ['cpu'],
    }
    if metadata['channel_order'] == 'none' and metadata['color_channel'] != 'gray':
        return False
    for key, allowed_values in pil_constraints.items():
        if key in metadata and metadata[key] not in allowed_values:
            return False
    return True


def pil_rgba_to_rgb(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image"):
        if validate_pil(source_metadata) and validate_pil(target_metadata):
            if (
                    source_metadata.get("color_channel") == "rgba"
                    and target_metadata.get("color_channel") == "rgb"
            ) or (
                    source_metadata.get("color_channel") == "graya"
                    and target_metadata.get("color_channel") == "gray"
            ):
                if is_differ_value_for_key(source_metadata, target_metadata, "color_channel"):
                    return (
                        "",
                        """def convert(var):
                            return var.convert("RGB") if var.mode == "RGBA" else var.convert("L")""",
                    )
    return None


def pil_convert_dtype(source_metadata, target_metadata) -> conversion:
    if are_both_same_data_repr(source_metadata, target_metadata, "PIL.Image") and is_differ_value_for_key(
            source_metadata, target_metadata, "data_type"):
        target_dtype = target_metadata.get("data_type")
        if target_dtype in ["uint8", "uint16", "float32", "int32"]:
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


pil_factories = [
    pil_rgba_to_rgb,
    pil_convert_dtype,
]
