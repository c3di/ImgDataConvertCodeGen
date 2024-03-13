from typing import Callable

from .code_generation import ConvertCodeGenerator
from .knowledge_graph_construction import get_knowledge_graph_constructor, Metadata, MetadataValues

_constructor = get_knowledge_graph_constructor()
_code_generator = ConvertCodeGenerator(_constructor.knowledge_graph)


def get_conversion(source_var_name: str, source_metadata: Metadata,
                   target_var_name: str, target_metadata: Metadata) -> str | None:
    """
    Generates Python code as a string that performs data conversion from a source variable to a target variable.

    Args:
        source_var_name (str): The name of the variable holding the source data.
        source_metadata (Metadata): A dictionary containing metadata about the source data.
            - `data_representation` (str): Description of data representation.
            - `color_channel` (str): Description of color channels.
            - `channel_order` (Literal['channel last', 'channel first', 'none']): Order of color channels.
            - `minibatch_input` (bool): Indicates if input is a minibatch.
            - `data_type` (str): Type of data ('uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'double'
              'float32(0to1)', 'float32(-1to1)', 'float64(0to1)', 'float64(-1to1)', 'double(0to1)', 'double(-1to1)'
              'int8', 'int16', 'int32', 'int64').
            - `device` (str): Device where the data is processed or stored.
        target_var_name (str): The name of the variable that will store the result of the conversion.
        target_metadata (Metadata): Metadata about the target data, structured similarly to source_metadata.

    Returns:
        str | None: A string containing the Python code necessary to perform the conversion, or None if conversion is not possible.

    Examples:
        >>> source_var_name = "source_image"
        >>> source_metadata = {"color_channel": "bgr", "channel_order": "channel last", ...}
        >>> target_var_name = "target_image"
        >>> target_metadata = {"color_channel": "rgb", "channel_order": "channel first", ...}
        >>> conversion_code = get_conversion(source_var_name, source_metadata, target_var_name, target_metadata)
        >>> print(conversion_code)
        # This example demonstrates converting an image from BGR color space with 'channel last' order to RGB color space with 'channel first' order.
        # The conversion code might look like this:
        # target_image = source_image[:, :, ::-1]  # Convert from BGR to RGB
        # target_image = target_image.transpose((2, 0, 1))  # Change from 'channel last' to 'channel first'
    """

    return _code_generator.get_conversion(source_var_name, source_metadata,
                                          target_var_name, target_metadata)


def get_convert_path(source_metadata: Metadata, target_metadata: Metadata):
    return _code_generator.get_convert_path(source_metadata, target_metadata)


def add_img_repr(new_metadata: MetadataValues):
    _constructor.add_new_metadata_values(new_metadata)
    _code_generator.knowledge_graph = _constructor.knowledge_graph


def add_convert_code_factory(new_factory: Callable | str):
    _constructor.add_new_edge_factory(new_factory)
    _code_generator.knowledge_graph = _constructor.knowledge_graph
