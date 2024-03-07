import re

import numpy as np
import pytest
import tensorflow as tf
import torch

from data_for_tests.image_data import get_test_image
from imgdataconvertcodegen import _code_generator


def is_image_equal(image1, image2):
    if type(image1) != type(image2):
        return False
    if isinstance(image1, np.ndarray):
        return np.array_equal(image1, image2)
    elif isinstance(image1, torch.Tensor):
        return torch.equal(image1, image2)
    # todo add other image types


def assert_exec_of_conversion_code_in_edge(edge, kg):
    source_metadata, target_metadata = edge
    edge_data = kg.get_edge_data(source_metadata, target_metadata)
    conversion = edge_data.get('conversion')
    assert conversion is not None, f"No conversion from {source_metadata} to {target_metadata}"
    assert len(conversion) == 2, (f"Expected two elements in the conversions, but got: {conversion} from"
                                  f" {source_metadata} to {target_metadata}")
    assert isinstance(conversion[0], str), (f"Expected the first element of the conversion to be a string, but got:"
                                            f" {conversion[0]} from {source_metadata} to {target_metadata}")
    assert isinstance(conversion[1], str), (f"Expected the second element of the conversion to be a string, but got:"
                                            f" {conversion[1]} from {source_metadata} to {target_metadata}")

    source_image = get_test_image(source_metadata)
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


def is_code_exec_on_cpu(edge):
    return edge[0]['device'] == 'cpu' and edge[1]['device'] == 'cpu'


def test_conversion_code_exec_on_cpu():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        if is_code_exec_on_cpu(edge):
            assert_exec_of_conversion_code_in_edge(edge, kg)


def is_on_gpu_as_data_repr(edge, data_reprs: list):
    if is_code_exec_on_cpu(edge):
        return False
    return edge[0]['data_representation'] in data_reprs and edge[1]['data_representation'] in data_reprs


def _check_pytorch_gpu_version_installed():
    return torch.version.cuda is not None


def pytorch_gpu_available():
    return torch.cuda.is_available() and _check_pytorch_gpu_version_installed()


def tensorflow_gpu_available():
    return len(tf.config.list_physical_devices('GPU')) > 0


@pytest.mark.skipif(not pytorch_gpu_available(),
                    reason="Test skipped because PyTorch is not installed with CUDA support or"
                           " no CUDA-compatible GPU is available.")
def test_conversion_code_exec_using_pytorch_gpu():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        if is_on_gpu_as_data_repr(edge, ['torch.tensor']):
            assert_exec_of_conversion_code_in_edge(edge, kg)


@pytest.mark.skipif(not tensorflow_gpu_available(),
                    reason="Test skipped because TensorFlow not configured for GPU acceleration or"
                           " no CUDA-compatible GPU is available.")
def test_conversion_code_exec_using_tensorflow_gpu():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        if is_on_gpu_as_data_repr(edge, ['tf.Tensor']):
            assert_exec_of_conversion_code_in_edge(edge, kg)


@pytest.mark.skipif(not tensorflow_gpu_available() or not pytorch_gpu_available(),
                    reason=f"Test skipped because {"TensorFlow" if not tensorflow_gpu_available() else "Pytorch"} not"
                           f" configured for GPU acceleration or no CUDA-compatible GPU is available.")
def test_conversion_code_exec_using_tensorflow_gpu_torch_gpu():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        if is_on_gpu_as_data_repr(edge, ['tf.Tensor', 'torch.tensor']):
            assert_exec_of_conversion_code_in_edge(edge, kg)
