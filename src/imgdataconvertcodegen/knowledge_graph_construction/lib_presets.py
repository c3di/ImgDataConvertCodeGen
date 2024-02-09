lib_presets = {
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


def get_available_libs():
    return list(lib_presets.keys())


def add_lib_preset(lib_name, lib_preset):
    lib_presets[lib_name] = lib_preset
