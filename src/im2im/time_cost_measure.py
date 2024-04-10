import re
import timeit
import math

from .test_image_util import random_test_image_and_expected
from .knowledge_graph_construction import encode_metadata


def time_cost(source, target, conversion, test_img_size=(256, 256), repeat_count=10):
    try:
        source_image, _ = random_test_image_and_expected(source, target, test_img_size)
    except Exception as e:
        return math.inf
    setup = f"{conversion[0]}\n{conversion[1]}"
    func_name = re.search(r'(?<=def )\w+', conversion[1]).group(0)
    code = f"actual_image = {func_name}(source_image)"
    try:
        execution_time = timeit.timeit(stmt=code, setup=setup, number=repeat_count, globals=locals())
    except Exception as e:
        raise RuntimeError(f'{e}, \ncode is {code}\nsetup is {setup}')
    return execution_time / repeat_count


def time_cost_in_kg(kg, test_img_size=(256, 256), repeat_count=10):
    time_costs = {}
    for edge in kg.edges:
        source, target = edge
        conversion = kg.get_edge_data(source, target).get('conversion')
        if conversion is not None:
            time_costs[(encode_metadata(source),
                        encode_metadata(target))] = time_cost(source, target, conversion, test_img_size, repeat_count)
    return time_costs
