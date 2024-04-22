"""
In the knowledge graph, the node is metadata description with some attributes for image data.
Some of the nodes are end nodes, which used to represent the image data in the specific libraries
while some of the nodes are intermediate nodes, which used to represent the intermediate data representation.
Typing the concrete metadata needs the effort to know exact value for each attribute. We introduce the metadata-mapper
to help user to get the metadata description for the image data in the specific library. Some function also supports
multiple format image data as the input.
"""

from typing import TypedDict, Literal, Optional

import numpy as np
import tensorflow as tf
import torch
from PIL import Image


class SourceImageDesc(TypedDict, total=False):
    lib: str
    image_dtype: Optional[Literal[
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32', 'int64',
        'float32(0to1)', 'float32(-1to1)',
        'float64(0to1)', 'float64(-1to1)',
        'double(0to1)', 'double(-1to1)'
    ]]

class ImageDesc(TypedDict, total=False):
    lib: str
    color_channel: Optional[Literal['gray', 'rgb', 'bgr', 'rgba', 'graya']]
    image_dtype: Optional[Literal[
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32', 'int64',
        'float32(0to1)', 'float32(-1to1)',
        'float64(0to1)', 'float64(-1to1)',
        'double(0to1)', 'double(-1to1)'
    ]]
    device: Optional[Literal['cpu', 'gpu']]

def end_metadata_mapper(source_image, source_image_desc: SourceImageDesc, target_image_desc: ImageDesc):
    mapper = {
        "numpy": (desc_to_metadata_numpy, image_to_desc_numpy),
        "scikit-image": (desc_to_metadata_scikit, image_to_desc_numpy),
        "opencv": (desc_to_metadata_opencv, image_to_desc_numpy),
        "scipy": (desc_to_metadata_scipy, image_to_desc_numpy),
        "matplotlib": (desc_to_metadata_matplotlib, image_to_desc_numpy),
        "PIL": (desc_to_metadata_pil, image_to_desc_pil),
        "torch": (desc_to_metadata_torch, image_to_desc_torch),
        "kornia": (desc_to_metadata_kornia, image_to_desc_torch),
        "tensorflow": (desc_to_metadata_tf, image_to_desc_tf),
    }
    if source_image_desc["lib"] not in mapper:
        raise ValueError(f"Unsupported library: {source_image_desc['lib']}. Supported libraries are {mapper.keys()}")
    if target_image_desc["lib"] not in mapper:
        raise ValueError(f"Unsupported library: {target_image_desc['lib']}. Supported libraries are {mapper.keys()}")

    source_metadata = mapper[source_image_desc['lib']][0]((mapper[source_image_desc['lib']][1](source_image, source_image_desc)))
    target_metadata = mapper[target_image_desc['lib']][0](target_image_desc, converted_from=source_metadata)

    return source_metadata, target_metadata


def desc_to_metadata_numpy(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                    'uint8')
    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)

    supported_color_channels = ["rgb", "gray"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for numpy.")

    supported_image_dtype = [
        'uint8', 'uint16', 'uint32',
        'int8', 'int16', 'int32',
        'float32(0to1)', 'float32(-1to1)',
        'float64(0to1)', 'float64(-1to1)',
        'double(0to1)', 'double(-1to1)',
    ]
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for numpy.")

    return {
        "data_representation": "numpy.ndarray",
        "color_channel": color_channel,
        "channel_order": "channel last" if color_channel != "gray" else "none",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }


def desc_to_metadata_scikit(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                    'uint8')
    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)

    if color_channel not in ["rgb", "gray"]:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are 'rgb' and 'gray' for scikit-image.")

    supported_image_dtype = [
        'uint8', 'uint16', 'uint32',
        "float32(0to1)", "float64(0to1)", "double(0to1)",
        "float32(-1to1)", "float64(-1to1)", "double(-1to1)",
        'int8', 'int16', 'int32'
    ]
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for scikit-image.")

    return {
        "data_representation": "numpy.ndarray",
        "color_channel": color_channel,
        "channel_order": "channel last" if color_channel != "gray" else "none",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }


def desc_to_metadata_opencv(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'bgr' if converted_from is None else converted_from.get('color_channel', 'bgr')
    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                    'uint8')

    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)

    if color_channel not in ["bgr", "gray"]:
        raise ValueError(f"Unsupported color channel: {color_channel}, "
                         f"Supported color channels are 'bgr' and 'gray for opencv-python.")

    supported_image_dtype = ['uint8']
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for opencv-python.")

    return {
        "data_representation": "numpy.ndarray",
        "color_channel": color_channel,
        "channel_order": "channel last" if color_channel != "gray" else "none",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }


def desc_to_metadata_scipy(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                    'uint8')
    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)

    supported_color_channels = ["rgb", "gray"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for scipy.")

    supported_image_dtype = ['uint8', 'uint16', 'float32(0to1)', 'int8', 'int16', "int32"]
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for scipy.")

    return {
        "data_representation": "numpy.ndarray",
        "color_channel": color_channel,
        "channel_order": "channel last" if color_channel != "gray" else "none",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }


def desc_to_metadata_matplotlib(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    color_channel = image_desc.get("color_channel", default_color_channel)
    supported_color_channels = ["rgb", "gray", "rgba"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for matplotlib.")

    if converted_from and converted_from.get("lib") == "PIL":
        return {
            "data_representation": "PIL.Image",
            "color_channel": color_channel,
            "channel_order": "none" if color_channel == "gray" else "channel last",
            "minibatch_input": False,
            "image_data_type": "uint8",
            "device": "cpu"
        }

    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                    'uint8')
    image_dtype = image_desc.get("image_dtype", default_image_dtype)
    supported_image_dtype = ['uint8', 'float32(0to1)', 'float64(0to1)']
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for matplotlib.")

    return {
        "data_representation": "numpy.ndarray",
        "color_channel": color_channel,
        "channel_order": converted_from.get(
            "channel_order") if converted_from else "none" if color_channel == "gray" else "channel last",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }

def desc_to_metadata_pil(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'uint8' if converted_from is None else converted_from.get('image_dtype',
                                                                                            'uint8')
    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)

    supported_color_channels = ["rgb", "gray", "rgba", "graya"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for PIL.")

    supported_image_dtype = ['uint8']
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for PIL.")

    return {
        "data_representation": "PIL.Image",
        "color_channel": color_channel,
        "channel_order": "channel last" if color_channel != "gray" else "none",
        "minibatch_input": False,
        "image_data_type": image_dtype,
        "device": "cpu"
    }


def desc_to_metadata_torch(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'float32(0to1)' if converted_from is None else converted_from.get('image_dtype',
                                                                                            'float32(0to1)')
    default_device = 'cpu' if converted_from is None else converted_from.get('device', 'cpu')

    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)
    device = image_desc.get("device", default_device)

    supported_color_channels = ["rgb", "gray"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for pytorch.")

    supported_image_dtype = [
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32(0to1)",
        "float64(0to1)",
        "double(0to1)",
    ]
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for pytorch.")

    supported_device = ['cpu', 'gpu']
    if device not in supported_device:
        raise ValueError(f"Unsupported device: {device}. Supported devices are {supported_device} for pytorch.")

    return {
        "data_representation": "torch.tensor",
        "color_channel": color_channel,
        "channel_order": "channel first",
        "minibatch_input": True,
        "image_data_type": image_dtype,
        "device": device
    }


def desc_to_metadata_kornia(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'float32(0to1)' if converted_from is None else converted_from.get('image_dtype',
                                                                                            'float32(0to1)')
    default_device = 'cpu' if converted_from is None else converted_from.get('device', 'cpu')

    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)
    device = image_desc.get("device", default_device)

    supported_color_channels = ["rgb", "gray"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for kornia.")

    supported_image_dtype = ['float32(0to1)']
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for kornia.")

    supported_device = ['cpu', 'gpu']
    if device not in supported_device:
        raise ValueError(f"Unsupported device: {device}. Supported devices are {supported_device} for kornia.")

    return {
        "data_representation": "torch.tensor",
        "color_channel": color_channel,
        "channel_order": "channel first",
        "minibatch_input": True,
        "image_data_type": image_dtype,
        "device": device
    }


def desc_to_metadata_tf(image_desc: ImageDesc, converted_from=None):
    default_color_channel = 'rgb' if converted_from is None else converted_from.get('color_channel', 'rgb')
    default_image_dtype = 'float32(0to1)' if converted_from is None else converted_from.get('image_dtype',
                                                                                            'float32(0to1)')
    default_device = 'cpu' if converted_from is None else converted_from.get('device', 'cpu')

    color_channel = image_desc.get("color_channel", default_color_channel)
    image_dtype = image_desc.get("image_dtype", default_image_dtype)
    device = image_desc.get("device", default_device)

    supported_color_channels = ["rgb", "gray"]
    if color_channel not in supported_color_channels:
        raise ValueError(f"Unsupported color channel: {color_channel}. "
                         f"Supported color channels are {supported_color_channels} for tensorflow.")

    supported_image_dtype = [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16(0to1)",
        "float32(0to1)",
        "float64(0to1)",
        "double(0to1)",
    ]
    if image_dtype not in supported_image_dtype:
        raise ValueError(f"Unsupported image data type: {image_dtype}. "
                         f"Supported image data types are {supported_image_dtype} for tensorflow.")

    supported_device = ['cpu', 'gpu']
    if device not in supported_device:
        raise ValueError(f"Unsupported device: {device}. Supported devices are {supported_device} for tensorflow.")

    return {
        "data_representation": "tf.tensor",
        "color_channel": color_channel,
        "channel_order": "channel last",
        "minibatch_input": True,
        "image_data_type": image_dtype,
        "device": device
    }


def image_to_desc_numpy(image: np.ndarray, source_image_desc: SourceImageDesc) -> ImageDesc:
    default_color_channel = "bgr" if source_image_desc.get("lib") == "opencv" else "rgb"
    if image.ndim == 2:
        color_channel = "gray"
    elif image.ndim == 3 and image.shape[-1] == 4:
        color_channel = "rgba"
    else:
        color_channel = default_color_channel

    return {
        "lib": source_image_desc["lib"],
        "color_channel": color_channel,
        "image_dtype": source_image_desc.get('image_dtype', 'uint8'),
        "device": "cpu"
    }


def image_to_desc_pil(image: Image.Image, source_image_desc: SourceImageDesc) -> ImageDesc:
    color_channel = {
        '1': 'gray',
        'L': 'gray',
        'LA': 'graya',
        'RGB': 'rgb',
        'RGBA': 'rgba',
    }.get(image.mode, 'rgb')
    return {
        "lib": source_image_desc["lib"],
        "color_channel": color_channel,
        "image_dtype": source_image_desc.get('image_dtype', 'uint8'),
        "device": "cpu"
    }


def image_to_desc_torch(image: torch.Tensor, source_image_desc: SourceImageDesc) -> ImageDesc:
    default_color_channel = "rgb"
    if image.ndim == 2:
        color_channel = "gray"
    elif image.ndim == 3 and image.shape[0] == 4:
        color_channel = "rgba"
    else:
        color_channel = default_color_channel
    return {
        "lib": source_image_desc["lib"],
        "color_channel": color_channel,
        "image_dtype": source_image_desc.get('image_dtype', 'float32(0to1)'),
        "device": 'gpu' if image.is_cuda else 'cpu'
    }


def image_to_desc_tf(image: tf.Tensor, source_image_desc: SourceImageDesc) -> ImageDesc:
    default_color_channel = "rgb"
    if image.ndim == 2:
        color_channel = "gray"
    elif image.ndim == 3 and image.shape[-1] == 4:
        color_channel = "rgba"
    else:
        color_channel = default_color_channel
    return {
            "lib": source_image_desc["lib"],
            "color_channel": color_channel,
            "image_dtype":source_image_desc.get('image_dtype', 'float32(0to1)'),
            "device": 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu'
        }
