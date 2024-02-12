import numpy as np
import torch
import pytest

from src.imgdataconvertcodegen import _code_generator


@pytest.fixture
def image_data():
    source_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    expected_image = torch.from_numpy(source_image).permute(2, 0, 1).unsqueeze(0)

    return {
        "source_image": source_image,
        "expected_image": expected_image
    }


def test_code_generation_using_metadata(image_data):
    source_image = image_data['source_image']
    target_var = 'target_result'

    source_metadata = {
        "data_representation": "numpy.ndarray",
        "color_channel": "rgb",
        "channel_order": "channel last",
        "minibatch_input": False,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    }
    target_metadata = {
        "data_representation": "torch.tensor",
        "color_channel": "rgb",
        "channel_order": "channel first",
        "minibatch_input": True,
        "data_type": "uint8",
        "intensity_range": "0to255",
        "device": "cpu"
    }

    # Prepare a custom scope that includes both global and local variables to ensure that the dynamically executed code
    # has access to necessary pre-defined variables and can also store new variables such as 'target_result'.
    # This is crucial in the pytest environment where test function scopes are isolated, and dynamically defined
    # variables might not be directly accessible due to Python's scoping rules.
    scope = globals().copy()
    scope.update(locals())

    convert_code = _code_generator.generate_code('source_image', source_metadata, target_var, target_metadata)
    exec(convert_code, scope)

    # Retrieve 'target_result' from the custom scope, ensuring accessibility despite the isolated test function scope
    target_result = scope.get(target_var)

    assert torch.equal(target_result, image_data['expected_image']), 'expected and actual images are not equal'


def test_code_generation_using_lib_names(image_data):
    source_image = image_data['source_image']
    target_var = 'target_result'

    # Prepare a custom scope that includes both global and local variables to ensure that the dynamically executed code
    # has access to necessary pre-defined variables and can also store new variables such as 'target_result'.
    # This is crucial in the pytest environment where test function scopes are isolated, and dynamically defined
    # variables might not be directly accessible due to Python's scoping rules.
    scope = globals().copy()
    scope.update(locals())

    convert_code = _code_generator.generate_code('source_image', 'numpy', target_var, 'torch')
    exec(convert_code, scope)

    # Retrieve 'target_result' from the custom scope, ensuring accessibility despite the isolated test function scope
    target_result = scope.get(target_var)

    assert torch.equal(target_result, image_data['expected_image']), 'expected and actual images are not equal'
