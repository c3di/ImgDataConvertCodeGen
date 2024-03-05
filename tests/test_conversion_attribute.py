import re

import numpy as np
import torch
import pytest

from imgdataconvertcodegen import _code_generator
from data_for_tests.image_data import get_test_image


def is_image_equal(image1, image2):
    if type(image1) != type(image2):
        return False
    if isinstance(image1, np.ndarray):
        return np.array_equal(image1, image2)
    elif isinstance(image1, torch.Tensor):
        return torch.equal(image1, image2)
    # todo add other image types


def check_no_cuda_gpu():
    """
    Check if no CUDA-enabled GPU is available.
    Returns:
        True if no GPU is available, False otherwise.
    """
    return not torch.cuda.is_available()


def edge_assertions(kg, edge, edge_data, conversion):
    assert conversion is not None, f"No conversion from {kg.get_node(edge[0])} to {kg.get_node(edge[1])}"
    assert len(conversion) == 2, (f"Expected two elements in the conversions, but got: {conversion} from"
                                  f" {kg.get_node(edge[0])} to {kg.get_node(edge[1])}")
    assert isinstance(conversion[0], str), (f"Expected the first element of the conversion to be a string, but got:"
                                            f" {conversion[0]} from {kg.get_node(edge[0])} to {kg.get_node(edge[1])}")
    assert isinstance(conversion[1], str), (
        f"Expected the second element of the conversion to be a string, but got:"
        f" {conversion[1]} from {kg.get_node(edge[0])} to {kg.get_node(edge[1])}")

    source_metadata = kg.get_node(edge[0])
    # print(f"source: {source_metadata}")
    source_image = get_test_image(source_metadata)
    target_metadata = kg.get_node(edge[1])
    # print(f"target: {target_metadata}")
    target_image = get_test_image(target_metadata)
    func_name = re.search(r'(?<=def )\w+', conversion[1]).group(0)

    scope = {}
    scope.update({'source_image': source_image})
    exec(f"""{conversion[0]}
{conversion[1]}
actual_image = {func_name}(source_image)""", scope)
    actual_image = scope.get('actual_image')

    assert is_image_equal(target_image, actual_image), (f"conversion from\n"
                                                        f"{source_metadata} to\n"
                                                        f"{target_metadata} failed\n"
                                                        f"using {conversion[1]} from\n"
                                                        f"{edge_data.get('factory')}")


@pytest.mark.skipif(check_no_cuda_gpu, reason="Test skipped: No Cuda GPU")
def test_conversion_property_of_edge():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        edge_data = kg.get_edge_data(edge[0], edge[1])
        conversion = edge_data.get('conversion')
        edge_assertions(kg, edge, edge_data, conversion)
