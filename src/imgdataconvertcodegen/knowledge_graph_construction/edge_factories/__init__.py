from .type import Conversion, EdgeFactory, FactoriesCluster, ConversionForMetadataPair
from .PIL import factories_cluster_for_pil, factories_for_pil_metadata_pair
from .numpy import factories_cluster_for_numpy
from .Pytorch import factories_cluster_for_Pytorch
from .Tensorflow import factories_cluster_for_tensorflow
from .inter_libs import (factories_cluster_for_numpy_pil, factories_cluster_for_numpy_torch,
                         factories_cluster_for_numpy_tensorflow)

factories_clusters: FactoriesCluster = [
    factories_cluster_for_pil,
    factories_cluster_for_numpy,
    factories_cluster_for_Pytorch,
    factories_cluster_for_tensorflow,
    factories_cluster_for_numpy_pil,
    factories_cluster_for_numpy_torch,
    factories_cluster_for_numpy_tensorflow,
]

factories_for_metadata_pair: ConversionForMetadataPair = factories_for_pil_metadata_pair
