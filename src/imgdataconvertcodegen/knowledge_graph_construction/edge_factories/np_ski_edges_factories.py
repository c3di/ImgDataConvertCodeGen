from .type import conversion
from ...metadata_differ import is_differ_value_for_key, is_data_type_and_intensity_range_differ


# this edge_factory is for methods with skimage and numpy

# https://stackoverflow.com/questions/21429261/array-conversion-using-scikit-image-from-integer-to-float
# returning None for double first convert with between_float64_double in default_edge_factories.py
def to_float32_or_64(source_metadata, target_metadata) -> conversion:
    source_dtype_str = source_metadata.get("data_type")
    target_dtype_str = target_metadata.get("data_type")
    # we have all types (also double) to not get ValueError (double -> None)
    dtype_mapping = {
        "uint8": "unsigned",
        "uint16": "unsigned",
        "uint32": "unsigned",
        "uint64": "unsigned",
        "int8": "signed",
        "int16": "signed",
        "int32": "signed",
        "int64": "signed",
        "float32": "float",
        "float64": "float",
        "double": "float"
    }

    # this part is only for speed (use between_float64_double in default_edge_factories.py first)
    if target_metadata.get("data_type") == "double":
        return None

    source_sign = dtype_mapping.get(source_dtype_str, None)
    if source_sign is None:
        raise ValueError(f"Unsupported source dtype: {source_dtype_str}")

    target_sign = dtype_mapping.get(target_dtype_str, None)
    if target_sign is None:
        raise ValueError(f"Unsupported target dtype: {target_dtype_str}")

    if (target_sign == "float" and source_sign == "unsigned" and
            target_metadata.get('intensity_range') == '-1to1'):
        if target_metadata.get("data_type") == "float32":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float32(var)"
        if target_metadata.get("data_type") == "float64":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float64(var)"

    if (target_sign == "float" and source_sign == "signed" and
            target_metadata.get('intensity_range') == '0to1'):
        if target_metadata.get("data_type") == "float32":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float32(var)"
        if target_metadata.get("data_type") == "float64":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float64(var)"

    if target_sign == "float" and source_sign == "float" and target_metadata.get('intensity_range') == 'full':
        if target_metadata.get("data_type") == "float32":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float32(var)"
        if target_metadata.get("data_type") == "float64":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_float64(var)"

    # for full intensity range and float as source metadata we can use both skimage and numpy,
    # because of order it will be used skimage
    if target_sign == "float" and target_metadata.get('intensity_range') == 'full':
        if target_metadata.get("data_type") == "float32":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import numpy as np", "def convert(var):\n  return var.astype(np.float32)"
        if target_metadata.get("data_type") == "float64":
            if is_data_type_and_intensity_range_differ(source_metadata, target_metadata):
                return "import numpy as np", "def convert(var):\n  return var.astype(np.float64)"

    return None


def ski_to_int16(source_metadata, target_metadata) -> conversion:
    if target_metadata.get('data_type') == 'int16':
        if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
            return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_int(var)"
    return None


def ski_to_uint8(source_metadata, target_metadata) -> conversion:
    if target_metadata.get('data_type') == 'uint8':
        if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
            return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_ubyte(var)"
    return None


def ski_to_uint16(source_metadata, target_metadata) -> conversion:
    if target_metadata.get('data_type') == 'uint16':
        if is_differ_value_for_key(source_metadata, target_metadata, "data_type"):
            return "import skimage as ski", "def convert(var):\n  return ski.util.img_as_uint(var)"
    return None


ski_factories = [
    to_float32_or_64,
    ski_to_int16,
    ski_to_uint8,
    ski_to_uint16
]
