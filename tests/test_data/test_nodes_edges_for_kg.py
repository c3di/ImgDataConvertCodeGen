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

# (source_id, target_id, (import_statement, conversion_function))
test_edges = [
    (1, 2, ("", "def convert(var):\n  return var[:, :, ::-1]")),
    (1, 3, ("import torch", "def convert(var):\n  return torch.from_numpy(var)")),
    (3, 4, ("", "def convert(var):\n  return var.permute(2, 0, 1)")),
]

new_edge = (4, 5, ("import torch", "def convert(var):\n  return torch.unsqueeze(var, 0)"))
