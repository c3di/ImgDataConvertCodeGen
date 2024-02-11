import re
import uuid


def create_unique_function(function_string):
    pattern = r'(def\s+)[a-zA-Z_]\w*(\()'
    new_name = f"cvt_{uuid.uuid4().hex}"
    new_function_string = re.sub(pattern, r'\1' + new_name + r'\2', function_string)
    return {"function_name": new_name, "function_definition": new_function_string}


def extract_func_body(code_str, argument, return_var_name):
    """
    Replaces a function call in the given code string with direct code execution using the specified argument,
    and assigns the result to the given return variable name.

    Args:
        code_str (str): The string containing the function definition and its call.
        argument (str): The actual argument to replace the function's parameter in the direct code execution.
        return_var_name (str): The name of the variable to which the result of the direct code execution is assigned.

    Returns:
        str: The modified code string with the function call replaced by direct code execution.

    Examples:
        >>> code_str = \"\"\"
        def convert(var):
            # Example transformation
            return var[:, :, ::-1]
        target_image = convert(source_image)
        \"\"\"
        >>> new_code_str = remove_function_call(code_str, 'source_image', 'target_image')
        >>> print(new_code_str)
        target_image = source_image[:, :, ::-1]

    Note:
        This function assumes there is only one function definition and one corresponding call in the input string.
    """
    function_pattern = re.compile(r'def .+\(([^)]+)\):\s*\n(.*?return\s+.*)', re.DOTALL | re.MULTILINE)
    match = function_pattern.search(code_str)
    if match:
        param_name, function_body = match.groups()
        escaped_param_name = re.escape(param_name)
        param_pattern = re.compile(rf'\b{escaped_param_name}\b')
        replaced_body = param_pattern.sub(argument, function_body)
        replaced_body_lines = replaced_body.strip().split('\n')
        replaced_body_lines[-1] = replaced_body_lines[-1].replace('return', f'{return_var_name} =', 1)

        adjusted_lines = [line[4:] if line.startswith('    ') else (line[1:] if line.startswith('\t') else line) for line in replaced_body_lines]
        return '\n'.join(adjusted_lines)
    return "No function body found."
