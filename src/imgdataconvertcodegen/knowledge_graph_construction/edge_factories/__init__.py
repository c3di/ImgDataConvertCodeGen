from .type import conversion
from .default_edges_factories import default_factories
from .Pytorch import factories_cluster_for_Pytorch
from .PIL import pil_factories
from .Tensforflow import factories_cluster_for_tensorflow
from .inter_libs import inter_libs_factories

all_edge_factories = default_factories + pil_factories + inter_libs_factories

factories_cluster = [
    factories_cluster_for_Pytorch,
    factories_cluster_for_tensorflow,
]