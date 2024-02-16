numpy_style_color = {
    "data_representation": "numpy.ndarray",
    "color_channel": "rgb",
    "channel_order": "channel last",
    "minibatch_input": False,
    "data_type": "uint8",
    "intensity_range": "full",
    "device": "cpu"
}

numpy_style_gray = {
    "data_representation": "numpy.ndarray",
    "color_channel": "gray",
    "channel_order": "none",
    "minibatch_input": False,
    "data_type": "uint8",
    "intensity_range": "full",
    "device": "cpu"
}

lib_presets = {
    "color": {
        "numpy": numpy_style_color,
        "scikit-image": numpy_style_color,
        "scipy": numpy_style_color,
        "matplotlib": numpy_style_color,
        'opencv-python': {
            "data_representation": "numpy.ndarray",
            "color_channel": "bgr",
            "channel_order": "channel last",
            "minibatch_input": False,
            "data_type": "uint8",
            "intensity_range": "full",
            "device": "cpu"
        },
        'PIL': {
            "data_representation": "PIL.Image",
            "color_channel": "rgb",
            "channel_order": "channel last",
            "minibatch_input": False,
            "data_type": "uint8",
            "intensity_range": "full",
            "device": "cpu"
        },
        "torch": {
            "data_representation": "torch.tensor",
            "color_channel": "rgb",
            "channel_order": "channel first",
            "minibatch_input": True,
            "data_type": "uint8",
            "intensity_range": "full",
            "device": "cpu"
        },
        "tensorflow": {
            "data_representation": "tf.tensor",
            "color_channel": "rgb",
            "channel_order": "channel last",
            "minibatch_input": True,
            "data_type": "uint8",
            "intensity_range": "full",
        }
    },
    "gray": {
        "numpy": numpy_style_gray,
        "scikit-image": numpy_style_gray,
        "opencv-python": numpy_style_gray,
        'scipy': numpy_style_gray,
        'matplotlib': numpy_style_gray,
        'PIL': {
            "data_representation": "PIL.Image",
            "color_channel": "gray",
            "channel_order": "none",
            "minibatch_input": False,
            "data_type": "uint8",
            "intensity_range": "full",
            "device": "cpu"
        },
        "torch": {
            "data_representation": "torch.tensor",
            "color_channel": "gray",
            "channel_order": "channel first",
            "minibatch_input": True,
            "data_type": "float32",
            "intensity_range": "normalized_unsigned",
            "device": "cpu"
        },
        "tensorflow": {
            "data_representation": "tf.tensor",
            "color_channel": "gray",
            "channel_order": "channel last",
            "minibatch_input": True,
            "data_type": "float32",
            "intensity_range": "normalized_unsigned",
        }
    }
}


def get_available_libs():
    return list(lib_presets.keys())


def get_lib_metadata(lib_name, color_channel):
    if color_channel not in lib_presets.keys():
        return None
    if lib_name not in lib_presets[color_channel].keys():
        return None
    return lib_presets[color_channel][lib_name]


def add_lib_preset(lib_name, lib_preset):
    lib_presets[lib_name] = lib_preset
