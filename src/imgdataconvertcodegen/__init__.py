import sys
from .api import get_conversion, get_convert_path, add_convert_code_factory, add_image_metadata, add_new_lib_preset
from .api import _code_generator, _builder
from .metadata_differ import *
from .function_util import *
from .measure import *
from .io import *


if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "ImgDataConvertCodeGen"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
