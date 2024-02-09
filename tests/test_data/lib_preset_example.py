lib_presets_example = {
    "numpy": {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    "scikit-image": {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    "torch": {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
        }
}
