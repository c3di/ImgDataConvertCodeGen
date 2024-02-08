## Conversion Graph
### Node structure
```python
image_data = {
    "data_representation": "numpy.ndarray",  # Options: numpy.ndarray, PIL.Image, torch.tensor, tf.tensor
    "color_channel": "rgb",                 # Options: rgb, gbr, gray, rgba, graya
    "channel_order": "channel last",        # Options: channel last, channel first
    "minibatch_input": False,               # Values: True, False
    "data_type": "uint8",                   # Options: uint8, uint16, uint32, float, float64, int8, int16, int32
    "intensity_range": "0to255",            # Options: 0to255, 0to1, -1to1
    "device": "cpu"                         # Options: cpu, gpu
}
```

More examples can be found in the `knowledge_graph_construction/default_nodes.py`.
### Edge numpy_bgr_to_rgb
```python
def numpy_bgr_to_rgb(source, target):
    def version_match():
        pass
        
    def metadata_match(source_metadata, target_metadata):
        pass
        
    if version_match() and metadata_match(source, target):
        return """
        def convert(value):
            return ....
        """
        
    return None
```

More examples can be found in the `knowledge_graph_construction/default_edge_factories.py`.
