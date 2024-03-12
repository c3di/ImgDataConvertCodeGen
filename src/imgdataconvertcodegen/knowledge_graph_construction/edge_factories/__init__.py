from .type import conversion
from .default_edges_factories import default_factories
from .Pytorch import pytorch_factories
from .PIL import pil_factories
from .Tensforflow import tensorflow_factories
from .inter_libs import inter_libs_factories

all_edge_factories = default_factories + pil_factories + pytorch_factories + tensorflow_factories + inter_libs_factories
