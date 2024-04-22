from typing import List, Union

from .code_generator import ConvertCodeGenerator
from .end_metadata_mapper import end_metadata_mapper, SourceImageDesc, ImageDesc
from .knowledge_graph_construction import (
    get_knowledge_graph_constructor,
    MetadataValues,
    FactoriesCluster,
    ConversionForMetadataPair,
    Metadata,
)

_constructor = get_knowledge_graph_constructor()
_code_generator = ConvertCodeGenerator(_constructor.knowledge_graph)


def im2im(source_image, source_image_desc:SourceImageDesc, tgt_im_desc: ImageDesc):
    """
    Converts an image from one format to another as specified by detailed descriptions of both source and target formats.

    Args:
        source_image: The image data to be converted, typically an array or tensor.
        source_image_desc (SourceImageDesc): A dictionary describing the format and characteristics of the source image, including:
            - **lib**: A string indicating the library associated with the source image. Supported libraries are "numpy", "scikit-image", "opencv", "scipy", "matplotlib", "PIL", "torch", "kornia", "tensorflow".
            - **image_dtype**: An optional attribute specifying the data type of the image pixels, which help determine how pixel data is interpreted. Possible values including
              'uint8', 'uint16', 'uint32', 'uint64',
              'int8', 'int16', 'int32', 'int64',
              'float32(0to1)', 'float32(-1to1)',
              'float64(0to1)', 'float64(-1to1)',
              'double(0to1)', 'double(-1to1)'.
        tgt_im_desc (ImageDesc): A dictionary providing a detailed description of the desired target image's format and characteristics, including:
            - **lib**: A string indicating the library associated with the target image. Supported libraries are "numpy", "scikit-image", "opencv", "scipy", "PIL", "torch", "kornia", "tensorflow".
            - **color_channel**: An optional attribute that specifies the color channel format of the target image, such as 'gray', 'rgb', 'bgr', 'rgba', or 'graya'.
            - **image_dtype**: An optional attribute that defines the data type of the image pixels, including
              'uint8', 'uint16', 'uint32', 'uint64',
              'int8', 'int16', 'int32', 'int64',
              'float32(0to1)', 'float32(-1to1)',
              'float64(0to1)', 'float64(-1to1)',
              'double(0to1)', 'double(-1to1)'.
            - **device**: An optional attribute that indicates the computing device ('cpu' or 'gpu') on which the image is stored or processed.

    Returns:
        Converted image data in the target format specified by tgt_im_desc.

    This function uses dynamic code execution to perform the conversion based on the descriptions provided. Care should be taken when defining the input descriptions to ensure compatibility and correct handling.
    """

    target_image_name = "target_image"
    code_str = im2im_code(source_image, "source_image", source_image_desc, target_image_name, tgt_im_desc)
    exec(code_str)
    return locals()[target_image_name]


def im2im_code(
        source_image,
        source_var_name: str,
        source_image_desc: SourceImageDesc,
        target_var_name: str,
        target_image_desc: ImageDesc,
) -> Union[str, None]:
    """
    Generates Python code as a string that performs data conversion from a source variable to a target variable based on specified descriptions.

    Args:
        source_image: The actual image data used for determining metadata in the conversion.
        source_var_name (str): The name of the variable holding the source image data.
        source_image_desc (SourceImageDesc): A dictionary describing the format and characteristics of the source image, including:
            - **lib**: A string indicating the library associated with the source image. Supported libraries are "numpy", "scikit-image", "opencv", "scipy", "matplotlib", "PIL", "torch", "kornia", "tensorflow".
            - **image_dtype**: An optional attribute specifying the data type of the image pixels, which help determine how pixel data is interpreted. Possible values including
              'uint8', 'uint16', 'uint32', 'uint64',
              'int8', 'int16', 'int32', 'int64',
              'float32(0to1)', 'float32(-1to1)',
              'float64(0to1)', 'float64(-1to1)',
              'double(0to1)', 'double(-1to1)'.
        target_var_name (str): The name of the variable that will store the converted image data.
        tgt_im_desc (ImageDesc): A dictionary providing a detailed description of the desired target image's format and characteristics, including:
            - **lib**: A string indicating the library associated with the target image. Supported libraries are "numpy", "scikit-image", "opencv", "scipy", "PIL", "torch", "kornia", "tensorflow".
            - **color_channel**: An optional attribute that specifies the color channel format of the target image, such as 'gray', 'rgb', 'bgr', 'rgba', or 'graya'.
            - **image_dtype**: An optional attribute that defines the data type of the image pixels, including
              'uint8', 'uint16', 'uint32', 'uint64',
              'int8', 'int16', 'int32', 'int64',
              'float32(0to1)', 'float32(-1to1)',
              'float64(0to1)', 'float64(-1to1)',
              'double(0to1)', 'double(-1to1)'.
            - **device**: An optional attribute that indicates the computing device ('cpu' or 'gpu') on which the image is stored or processed.

    Returns:
        str | None: A string containing Python code necessary for performing the image conversion, or None if conversion is not feasible given the provided descriptions.

    Examples:
        >>> source_image_desc = {"lib": "numpy", "image_dtype": "float32(0to1)"}
        >>> target_image_desc = {"lib": "torch", "image_dtype": "float32(0to1)", "device": "cpu"}
        >>> source_image = np.random.rand(10, 10, 3)
        >>> conversion_code = im2im_code(source_image, "source_image", source_image_desc, "target_image", target_image_desc)
        >>> print(conversion_code)
        # Example output might be:
        # import torch
        # target_image = torch.from_numpy(source_image)
        # target_image = target_image.permute(2, 0, 1)  # Convert HWC to CHW format expected by PyTorch.
    Notes:
        The function uses an intermediary mapping through 'end_metadata_mapper' to determine the appropriate metadata conversions between source and target formats. This mapping influences the generated code.
    """

    source_metadata, target_metadata = end_metadata_mapper(source_image, source_image_desc, target_image_desc)
    return im2im_code_by_metadata(
        source_var_name, source_metadata, target_var_name, target_metadata
    )


def im2im_code_by_metadata(
        source_var_name: str,
        source_metadata: Metadata,
        target_var_name: str,
        target_metadata: Metadata,
) -> Union[str, None]:
    return _code_generator.get_conversion(
        source_var_name, source_metadata, target_var_name, target_metadata
    )


def im2im_path(source_image, source_image_desc: SourceImageDesc, target_image_desc: ImageDesc):
    source_metadata, target_metadata = end_metadata_mapper(
        source_image, source_image_desc, target_image_desc
    )
    return im2im_path_by_metadata(source_metadata, target_metadata)


def im2im_path_by_metadata(source_metadata: Metadata, target_metadata: Metadata):
    return _code_generator.get_convert_path(source_metadata, target_metadata)


def config_astar_goal_function(cpu_penalty: float, gpu_penalty: float,
                               include_time_cost: bool = False, test_img_size=(256, 256)):
    """
    We use A* to find the shortest path in the knowledge graph. The goal function is only step cost by default.
    You can use this function to set `cpu penalty`, `gpu penalty`  or enable `execution time cost` to the goal function.
    """
    _code_generator.config_astar_goal_function(cpu_penalty, gpu_penalty, include_time_cost, test_img_size)


def add_meta_values_for_image(new_metadata: MetadataValues):
    _constructor.add_metadata_values(new_metadata)
    _code_generator.knowledge_graph = _constructor.knowledge_graph


def add_edge_factory_cluster(factory_cluster: FactoriesCluster):
    _constructor.add_edge_factory_cluster(factory_cluster)
    _code_generator.knowledge_graph = _constructor.knowledge_graph


def add_conversion_for_metadata_pairs(
        pairs: Union[List[ConversionForMetadataPair], ConversionForMetadataPair]
):
    _constructor.add_conversion_for_metadata_pairs(pairs)
    _code_generator.knowledge_graph = _constructor.knowledge_graph
