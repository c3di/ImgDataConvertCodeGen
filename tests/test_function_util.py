import re
from unittest.mock import patch
from src.imgdataconvertcodegen.function_util import create_unique_function


def test_create_unique_function_name_replacement():
    function_string = """
def original_function_name(a, b):
    return a + b
"""

    expected_new_name_prefix = "cvt_"

    with patch('src.imgdataconvertcodegen.function_util.uuid') as mock_uuid:

        mock_uuid.uuid4.return_value.hex = "1234567890abcdef"
        expected_uuid = "1234567890abcdef"
        expected_new_name = expected_new_name_prefix + expected_uuid

        result = create_unique_function(function_string)

        assert "function_name" in result
        assert "function_definition" in result
        assert result["function_name"] == expected_new_name

        new_function_definition = result["function_definition"]
        pattern = rf'def\s+{expected_new_name}\('
        assert re.search(pattern, new_function_definition), "New function name not found in the function definition"
        assert "return a + b" in new_function_definition, "The body of the original function has been altered"
