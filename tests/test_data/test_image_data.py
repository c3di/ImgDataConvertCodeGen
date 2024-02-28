import numpy as np
import torch
import tensorflow as tf

h, w = 20, 20
r = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
g = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
b = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b


def get_test_image(metadata, target):
    if metadata["data_representation"] == "numpy.ndarray":
        return get_numpy_image(metadata, target)
    elif metadata["data_representation"] == "torch.tensor":
        return get_torch_image(metadata, target)
    elif metadata["data_representation"] == "tf.tensor":
        return get_tensorflow_image(metadata, target)
    else:
        # todo add other data representations
        raise ValueError(
            f"Unsupported data representation: {metadata['data_representation']}"
        )


def get_numpy_image(metadata, target):
    img = None
    img = np.stack([r, g, b], axis=-1) if metadata["color_channel"] == "rgb" else np.stack([b, g, r], axis=-1) if metadata["color_channel"] == "bgr" else gray

    if target:
        target_dtype = np.dtype(metadata["data_type"])
        img = img.astype(target_dtype)

    return img


def get_torch_image(metadata, target):
    img = None
    if metadata["color_channel"] == "rgb":
        img = torch.stack(
            [torch.from_numpy(r), torch.from_numpy(g), torch.from_numpy(b)], dim=-1
        )
    elif metadata["color_channel"] == "bgr":
        img = torch.stack(
            [torch.from_numpy(b), torch.from_numpy(g), torch.from_numpy(r)], dim=-1
        )
    elif metadata["color_channel"] == "gray":
        img = torch.from_numpy(gray)
        img = img.unsqueeze(-1)

    if target:
        target_type = getattr(torch, metadata["data_type"])
        img = img.to(dtype = target_type)
        if target_dtype == 'float'
            img = image/255.0

    if metadata["channel_order"] == "channel first":
        img = img.permute(2, 0, 1)

    if metadata["minibatch_input"]:
        img = img.unsqueeze(0)

    if metadata["device"] == "gpu":
        img = img.cuda()
    return img


def get_tensorflow_image(metadata, target):
    img = None
    if metadata["color_channel"] == "rgb":
        img = tf.stack(
            [
                tf.convert_to_tensor(r, dtype=tf.uint8),
                tf.convert_to_tensor(g, dtype=tf.uint8),
                tf.convert_to_tensor(b, dtype=tf.uint8),
            ],
            axis=-1,
        )
    elif metadata["color_channel"] == "bgr":
        img = tf.stack(
            [
                tf.convert_to_tensor(b, dtype=tf.uint8),
                tf.convert_to_tensor(g, dtype=tf.uint8),
                tf.convert_to_tensor(r, dtype=tf.uint8),
            ],
            axis=-1,
        )
    elif metadata["color_channel"] == "gray":
        img = tf.convert_to_tensor(gray, dtype=tf.uint8)
        img = tf.expand_dims(img, -1)

    if target:
        target_type = tf.dtypes.as_dtype(metadata["data_type"])
        img = tf.cast(img, target_type)
        if target_type == 'td.float32':
            img = img / 255.0

    if metadata["channel_order"] == "channel first":
        img = tf.transpose(img, perm=[2, 0, 1])

    if metadata["minibatch_input"]:
        img = tf.expand_dims(img, 0)

    return img
