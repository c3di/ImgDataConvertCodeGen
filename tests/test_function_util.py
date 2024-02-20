from imgdataconvertcodegen.util import extract_func_body


def test_remove_intermediate_functon_call():
    code_str = """
def convert(var):
    # transformation
    var_with_var_in_name = var + 1
    for i in range(len(var)):
        var[i] = var[i] * 2  # Indentation inside the loop
    return var
    """

    expected_output = """# transformation
var_with_var_in_name = source_image + 1
for i in range(len(source_image)):
    source_image[i] = source_image[i] * 2  # Indentation inside the loop
target_image = source_image
    """.strip()

    actual = extract_func_body(code_str, 'source_image', 'target_image').strip()

    assert actual == expected_output, "The transformed code does not match the expected output."


def test_extract_func_body_fail():
    code_str = """
    # transformation
    var_with_var_in_name = var + 1
    for i in range(len(var)):
        var[i] = var[i] * 2  # Indentation inside the loop
    return var
    """

    actual = extract_func_body(code_str, 'source_image', 'target_image')

    assert actual is None, ("The function should return None if the function definition is not found in the input "
                            "string.")
