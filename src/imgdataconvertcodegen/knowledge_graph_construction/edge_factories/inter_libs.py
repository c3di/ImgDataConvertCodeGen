from .PIL import is_attribute_value_valid_for_pil, is_metadata_valid_for_pil
from .Pytorch import is_attribute_value_valid_for_torch, is_metadata_valid_for_torch
from .Tensorflow import is_attribute_value_valid_for_tensorflow, is_metadata_valid_for_tensorflow
from .numpy import is_attribute_value_valid_for_numpy, is_metadata_valid_for_numpy
from .type import Conversion, FactoriesCluster


def numpy_to_torch(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "numpy.ndarray"
            and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return (
            "import torch",
            "def convert(var):\n  return torch.from_numpy(var)",
        )
    return None


def torch_to_numpy(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "torch.tensor"
            and target_metadata.get("data_representation") == "numpy.ndarray"
    ):
        return (
            "import torch",
            "def convert(var):\n  return var.numpy(force=True)",
        )
    return None


def is_convert_between_numpy_and_torch(source_metadata, target_metadata):
    # only one attribute (data representation) change, so we only check the source_metadata
    return (
            is_attribute_value_valid_for_torch(source_metadata) and
            is_attribute_value_valid_for_numpy(source_metadata) and
            is_metadata_valid_for_torch(source_metadata) and
            is_metadata_valid_for_numpy(source_metadata)
    )


factories_cluster_for_numpy_torch: FactoriesCluster= (
    is_convert_between_numpy_and_torch,
    [
        numpy_to_torch,
        torch_to_numpy,
    ]
)


def numpy_to_pil(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "numpy.ndarray"
            and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return (
            "from PIL import Image",
            "def convert(var):\n  return Image.fromarray(var)",
        )
    return None


def pil_to_numpy(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "PIL.Image"
            and target_metadata.get("data_representation") == "numpy.ndarray"
    ):
        return (
            "import numpy as np",
            "def convert(var):\n  return np.array(var)",
        )
    return None


def is_convert_between_numpy_and_pil(source_metadata, target_metadata):
    # only one attribute (data representation) change, so we only check the source_metadata
    return (
            is_attribute_value_valid_for_pil(source_metadata) and
            is_attribute_value_valid_for_numpy(source_metadata) and
            is_metadata_valid_for_pil(source_metadata) and
            is_metadata_valid_for_numpy(source_metadata)
    )


factories_cluster_for_numpy_pil: FactoriesCluster = (
    is_convert_between_numpy_and_pil,
    [
        numpy_to_pil,
        pil_to_numpy,
    ]
)


def tensorflow_to_numpy(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "tf.tensor"
            and target_metadata.get("data_representation") == "numpy.ndarray"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.make_ndarray(var)",
        )
    return None


def numpy_to_tensorflow(source_metadata, target_metadata) -> Conversion:
    if (
            source_metadata.get("data_representation") == "numpy.ndarray"
            and target_metadata.get("data_representation") == "tf.tensor"

    ):
        return (
            "import tensorflow as tf",
            f"""def convert(var):
    dtype = getattr(tf, '{source_metadata['image_data_type']}')
    return tf.convert_to_tensor(var, dtype = dtype)""",
        )
    return None


def is_convert_between_numpy_and_tensorflow(source_metadata, target_metadata):
    # only one attribute (data representation) change, so we only check the source_metadata
    return (
            is_attribute_value_valid_for_tensorflow(source_metadata) and
            is_attribute_value_valid_for_numpy(source_metadata) and
            is_metadata_valid_for_tensorflow(source_metadata) and
            is_metadata_valid_for_numpy(source_metadata)
    )


factories_cluster_for_numpy_tensorflow: FactoriesCluster = (
    is_convert_between_numpy_and_tensorflow,
    [
        numpy_to_tensorflow,
        tensorflow_to_numpy,
    ]
)
