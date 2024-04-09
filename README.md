# imgdataconvertcodegen

[![PyPI - Version](https://img.shields.io/pypi/v/imgdataconvertcodegen.svg)](https://pypi.org/project/imgdataconvertcodegen/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ImgDataConvertCodeGen)](https://pypi.org/project/imgdataconvertcodegen/)
[![Documentation](https://img.shields.io/badge/Doc-ReadMe-blue)](https://github.com/c3di/ImgDataConvertCodeGen/blob/main/README.md)
[![Tests](https://github.com/c3di/ImgDataConvertCodeGen/actions/workflows/python%20tests%20with%20coverage.yml/badge.svg)](https://github.com/c3di/ImgDataConvertCodeGen/actions/workflows/python%20tests%20with%20coverage.yml)
[![codecov](https://codecov.io/github/c3di/ImgDataConvertCodeGen/graph/badge.svg?token=BWBXANX8W7)](https://codecov.io/github/c3di/ImgDataConvertCodeGen)

The `imgdataconvertcodegen` package offers an automated approach to generate conversion code for in-memory image representations, such as `numpy.ndarray`, `torch.tensor`, `PIL.Image`, and others using a knowledge graph of data types.

At the core of the package is a knowledge graph in which nodes represent data types and edges represent conversion code snippets between the data types. By traversing the path from source to target data types within the graph, the package collect each conversion code snippet along the path to generate the final conversion code.


## Installation

Install the package via pip:
```bash
pip install imgdataconvertcodegen
```
## Usage

One example from the image data in numpy to the image data in PyTorch:
```python
from imgdataconvertcodegen import get_conversion_code

source_image_desc = {"lib": "numpy"}
target_image_desc = {"lib": "torch", "image_dtype": "uint8"}
code = get_conversion_code("source_image", source_image_desc, "target_image", target_image_desc)
```
The generated conversion code will be as follows:
```python
import torch
image = torch.from_numpy(source_image)
image = image.permute(2, 0, 1)
target_image = torch.unsqueeze(image, 0)
```

## Evaluation


**Accuracy**

All primitive conversion code snippets are stored within the edges of the knowledge graph. These snippets are verified through execution checks to guarantee their correctness. For a more in-depth examination, please refer to the [`test_conversion_code_in_kg.py`](https://github.com/c3di/ImgDataConvertCodeGen/blob/main/tests/test_conversion_code_in_kg.py).

**Performance profiling**

The performance of knowledge graph construction and code generation processes is meticulously analyzed using the cProfile module. For comprehensive insights, please refer to the [`profiling notebooks`](https://github.com/c3di/ImgDataConvertCodeGen/blob/main/profile).

**Usability Evaluation**

Please refer to [Usability Evaluation](https://github.com/c3di/ImgDataConvertCodeGen_Evaluation).

## Development


For detailed instructions on developing, building, and publishing this package, please refer to the [README_DEV](https://github.com/c3di/ImgDataConvertCodeGen/blob/main/README_Dev.md).


## Cite

if you use our tool or code in your research, please cite the following paper:

Todo

## License

This project is licensed under the MIT License. See the LICENSE file for details.
