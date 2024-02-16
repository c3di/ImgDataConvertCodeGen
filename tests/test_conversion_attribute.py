import re
import numpy as np
import torch
from test_data.test_image_data import get_test_image

from src.imgdataconvertcodegen import _code_generator


def assert_image_equal(image1, image2):
    assert type(image1) == type(image2), f"image1 type: {type(image1)}, image2 type: {type(image2)}"
    if isinstance(image1, np.ndarray):
        assert np.array_equal(image1, image2), f"image1: {image1}, image2: {image2} are not equal"
    elif isinstance(image1, torch.Tensor):
        assert torch.equal(image1, image2), f"image1: {image1}, image2: {image2} are not equal"
    # todo add other image types


def test_conversion_property_of_edge():
    kg = _code_generator.knowledge_graph
    for edge in kg.edges:
        conversion = kg.get_edge(edge[0], edge[1]).get('conversion')
        assert conversion is not None, f"No conversion from {kg.get_node(edge[0])} to {kg.get_node(edge[1])}"
        assert len(conversion) == 2, (f"Expected two elements in the conversions, but got: {conversion} from"
                                      f" {kg.get_node(edge[0])} to {kg.get_node(edge[1])}")

        source_metadata = kg.get_node(edge[0])
        source_image = get_test_image(source_metadata)
        target_metadata = kg.get_node(edge[1])
        target_image = get_test_image(target_metadata)
        func_name = re.search(r'(?<=def )\w+', conversion[1]).group(0)

        scope = {}
        scope.update({'source_image': source_image})
        exec(f"""{conversion[0]}
{conversion[1]}
actual_image = {func_name}(source_image)""", scope)
        actual_image = scope.get('actual_image')

        assert_image_equal(target_image, actual_image), f"conversion from {source_metadata} to {target_metadata} failed"
