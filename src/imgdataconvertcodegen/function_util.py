import re
import uuid


def create_unique_function(function_string):
    pattern = r'(def\s+)[a-zA-Z_]\w*(\()'
    new_name = f"cvt_{uuid.uuid4().hex}"
    new_function_string = re.sub(pattern, r'\1' + new_name + r'\2', function_string)
    return {"function_name": new_name, "function_definition": new_function_string}
