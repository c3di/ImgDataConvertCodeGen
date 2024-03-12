from .type import conversion


def pil_to_torch(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "PIL.Image"
            and target_metadata.get("data_representation") == "torch.tensor"
    ):
        # todo: https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
        common_constraints = {
            "color_channel": ['rgb', 'gray'],
            "minibatch_input": [False],
            "data_type": ['uint8'],
            "device": ['cpu']
        }

        for key, allowed_values in common_constraints.items():
            if not source_metadata.get(key) in allowed_values:
                return None

        return (
            "from torchvision.transforms import ToTensor",
            "def convert(var):\n  return ToTensor()(var)",
        )
        return None


# TODO: https://pytorch.org/vision/master/generated/torchvision.transforms.functional.to_pil_image.html
def torch_to_pil(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "torch.tensor"
            and target_metadata.get("data_representation") == "PIL.Image"
    ):

        common_constraints = {
            "color_channel": ['rgb', 'gray'],
            "channel_order": ['channel last', 'channel first', 'none'],
            "minibatch_input": [False],
            "data_type": ['uint8'],
            "device": ['cpu']
        }

        for key, allowed_values in common_constraints.items():
            if not source_metadata.get(key) in allowed_values:
                return None
        return (
            "from torchvision.transforms import ToPILImage",
            "def convert(var):\n  return ToPILImage()(var)",
        )
    return None


# TODO: split to  PIL to numpy and numpy to tensor
# GPU？
def pil_to_tf(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "PIL.Image"
            and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return (
            "import tensorflow as tf\nimport numpy as np",
            """def convert(var):
        np_array = np.array(var)
        return tf.convert_to_tensor(np_array, dtype=tf.uint8)""",
        )
    return None


# TODO: split to tf-to-numpy and numpy-to-tensor
# GPU？
def tf_to_pil(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "tf.tensor"
            and target_metadata.get("data_representation") == "PIL.Image"
    ):
        return (
            "from PIL import Image",
            "def convert(var):\n  return Image.fromarray(var.numpy())",
        )
    return None


# TODO: split to numpy-to-tf and tf-to-torch
# GPU？
def torch_to_tf(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "torch.tensor"
            and target_metadata.get("data_representation") == "tf.tensor"
    ):
        return (
            "import tensorflow as tf",
            "def convert(var):\n  return tf.convert_to_tensor(var.numpy(), dtype=tf.as_dtype(var.dtype))",
        )
    return None


# TODO: numpy-to-torch and torch-to-numpy
def tf_to_torch(source_metadata, target_metadata) -> conversion:
    if (
            source_metadata.get("data_representation") == "tf.tensor"
            and target_metadata.get("data_representation") == "torch.tensor"
    ):
        return (
            "import torch",
            "def convert(var):\n  return torch.tensor(var_np = var.numpy(), dtype=var.dtype.as_numpy_dtype)",
        )
    return None


inter_libs_factories = [
    pil_to_torch,
    pil_to_tf,
    torch_to_pil,
    torch_to_tf,
    tf_to_pil,
    tf_to_torch
]
