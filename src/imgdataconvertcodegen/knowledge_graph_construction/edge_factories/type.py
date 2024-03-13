from typing import Callable, List, Tuple
from ..metedata import Metadata

import_code = str
convert_code = str
Conversion = Tuple[import_code, convert_code] | None
EdgeFactory = Callable[[Metadata, Metadata], Conversion]
FactoriesCluster = Tuple[Callable[[Metadata, Metadata], bool], List[EdgeFactory]]
ConversionForMetadataPair = Tuple[Metadata, Metadata, Conversion]
