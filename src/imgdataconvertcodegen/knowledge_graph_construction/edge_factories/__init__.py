from .type import conversion
from .PIL import factories_cluster_for_pil
from .Pytorch import factories_cluster_for_Pytorch
from .Tensorflow import factories_cluster_for_tensorflow
from .inter_libs import (factories_cluster_for_numpy_pil, factories_cluster_for_numpy_torch,
                         factories_cluster_for_numpy_tensorflow)

#Todo: remove and change in the knowledge graph construction
all_edge_factories = []

factories_cluster = [
    factories_cluster_for_pil,
    factories_cluster_for_Pytorch,
    factories_cluster_for_tensorflow,
    factories_cluster_for_numpy_pil,
    factories_cluster_for_numpy_torch,
    factories_cluster_for_numpy_tensorflow,
]