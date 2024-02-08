def is_single_metadata_differ(src_metadata, dest_metadata):
    differences = 0
    all_keys = set(src_metadata) | set(dest_metadata)
    for key in all_keys:
        if src_metadata.get(key) != dest_metadata.get(key):
            differences += 1
            if differences > 1:
                return False
    return differences == 1


nodes = [
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "-1to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "-1to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "-1to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "-1to1",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "gray",
        "channel_order": "none",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "rgba",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "graya",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "graya",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "graya",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "PIL.Image",
        "color_channel": "graya",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "torch.tensor",
        "color_channel": "gray",
        "channel_order": "channel first",
        "minibatch_input": False,
        "data_type": "double",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": True,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float32",
        "intensity_range": "0to1",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "float64",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int8",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int16",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int32",
        "intensity_range": "0to255",
        "device": "gpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "cpu"
    },
    {
        "data_representation": "tf.tensor",
        "color_channel": "gray",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "int64",
        "intensity_range": "0to255",
        "device": "gpu"
    }
]
