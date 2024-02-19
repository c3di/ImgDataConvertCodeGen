import numpy as np
import torch

h, w = 20, 20
r = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
g = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
b = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b


def get_test_image(metadata):
    if metadata["data_representation"] == "numpy.ndarray":
        return get_numpy_image(metadata)
    elif metadata["data_representation"] == "torch.tensor":
        return get_torch_image(metadata)
    else:
        # todo add other data representations
        raise ValueError(f"Unsupported data representation: {metadata['data_representation']}")


def get_numpy_image(metadata):
    img = None
    if metadata["color_channel"] == "rgb":
        img = np.stack([r, g, b], axis=-1)
    elif metadata["color_channel"] == "bgr":
        img = np.stack([b, g, r], axis=-1)
    else:
        img = gray
    # todo data type from uint8 to target data type
    # todo need to normalize the image?
    return img


def get_torch_image(metadata):
    img = None
    if metadata["color_channel"] == "rgb":
        img = torch.stack([torch.from_numpy(r), torch.from_numpy(g), torch.from_numpy(b)], dim=-1)
    elif metadata["color_channel"] == "bgr":
        img = torch.stack([torch.from_numpy(b), torch.from_numpy(g), torch.from_numpy(r)], dim=-1)
    elif metadata["color_channel"] == "gray":
        img = torch.from_numpy(gray)
        img = img.unsqueeze(-1)

    if metadata["channel_order"] == "channel first":
        img = img.permute(2, 0, 1)

    if metadata["minibatch_input"]:
        img = img.unsqueeze(0)

    # todo data type from uint8 to target data type
    # todo need to normalize the image?

    if metadata["device"] == "gpu":
        img = img.cuda()

    return img
