from importlib.metadata import PackageNotFoundError, version

from .interface_py_api import _code_generator, _constructor
from .interface_py_api import (im2im, im2im_code, im2im_path, add_edge_factory_cluster,
                               add_meta_values_for_image, add_conversion_for_metadata_pairs,
                               config_astar_goal_function)
from .knowledge_graph_construction import find_closest_metadata
from .test_image_util import is_image_equal, random_test_image_and_expected
from .time_cost_measure import time_cost, time_cost_in_kg
from .metadata_differ import *

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "im2im"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
