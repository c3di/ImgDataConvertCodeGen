from .type import conversion
from .default_edges_factories import default_factories
from .PILTorchTensor_edges_factories import pil_factories

all_edge_factories = default_factories + pil_factories
