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
        "intensity_range": "0-255",
        "device": "cpu",
    },
    {
        "data_representation": "numpy.ndarray",
        "color_channel": "gbr",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0-255",
        "device": "cpu",
    },
    {
    #...
    }
    # Todo... add more nodes from the table
]
