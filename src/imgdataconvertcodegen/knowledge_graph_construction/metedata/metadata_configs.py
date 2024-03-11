from .type import PossibleValuesForImgRepr
from .util import add_img_metadata_config


def is_valid_metadata_tensor(values: PossibleValuesForImgRepr) -> bool:
    return True


def add_default_img_metadata_config():
    add_img_metadata_config("torch.tensor", {
        "color_channel": ['rgb', 'gray'],
        "channel_order": ['channel first'],
        "minibatch_input": [True, False],
        "data_type": ['uint8', 'float32', 'float64', 'double',
                      'int8', 'int16', 'int32', 'int64'],
        "intensity_range": ['full', 'normalized_unsigned'],
        "device": ['cpu', 'gpu']
    }, is_valid_metadata_tensor)

    # todo: add more default configurations

    # "numpy.ndarray": {
    #     "color_channel": ['rgb', 'bgr', 'gray'],
    #     "channel_order": ['channel last', 'none'],
    #     "minibatch_input": [False],
    #     "data_type": ['uint8', 'uint16', 'uint32',
    #                   'float', 'int8', 'int16', 'int32'],
    #     "intensity_range": ['full', 'normalized_unsigned', 'normalized_signed'],
    #     "device": ['cpu']
    # },
    # "PIL.Image": {
    #     "color_channel": ['rgb', 'gray', 'rgba', 'graya'],
    #     "channel_order": ['channel last', 'none'],
    #     "minibatch_input": [False],
    #     "data_type": ['uint8'],
    #     "intensity_range": ['full'],
    #     "device": ['cpu']
    # },
    # "tf.tensor": {
    #     "color_channel": ['rgb', 'gray'],
    #     "channel_order": ['channel last'],
    #     "minibatch_input": [True, False],
    #     "data_type": ['uint8', 'uint16', 'uint32', 'uint64',
    #               'float16', 'float32', 'float64',
    #               'int8', 'int16', 'int32', 'int64'],
    #     "intensity_range": ['full', 'normalized_unsigned'],
    #     "device": ['cpu', 'gpu']
    # }
