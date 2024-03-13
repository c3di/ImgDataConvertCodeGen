from .type import MetadataValues, Metadata
from .util import *


metadata_values: MetadataValues = {
    "data_representation": ["torch.tensor", "numpy.ndarray", "PIL.Image", "tf.tensor"],
    "color_channel": ['rgb', 'bgr', 'gray', 'rgba', 'graya'],
    "channel_order": ['channel last', 'channel first', 'none'],
    "minibatch_input": [True, False],
    "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
                  'float16', 'float32', 'float64', 'double',
                  'int8', 'int16', 'int32', 'int64'],
    # intensity_range can be also -1 to 1 or 0 to 1, https://scikit-image.org/docs/stable/user_guide/data_types.html
    "intensity_range": ['full', '0to1', '-1to1'],
    "device": ['cpu', 'gpu']
}
