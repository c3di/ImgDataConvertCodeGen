test_nodes = [
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "full",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "bgr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "full",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "full",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "full",
        "device": "cpu"
    }

]

new_node = {
    "data_representation": "torch.tensor",
    "color_channel": "rgb",
    "channel_order": "channel first",
    "minibatch_input": True,
    "data_type": "uint8",
    "intensity_range": "full",
    "device": "cpu"
}

test_edges = [
    (test_nodes[0], test_nodes[1], ("", "def convert(var):\n  return var[:, :, ::-1]")),
    (test_nodes[0], test_nodes[2], ("import torch", "def convert(var):\n  return torch.from_numpy(var)")),
    (test_nodes[2], test_nodes[3], ("", "def convert(var):\n  return var.permute(2, 0, 1)")),
]

new_edge = (test_nodes[3], new_node, ("import torch", "def convert(var):\n  return torch.unsqueeze(var, 0)"))
