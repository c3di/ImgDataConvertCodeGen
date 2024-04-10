# im2im

[![PyPI - Version](https://img.shields.io/pypi/v/im2im.svg)](https://pypi.org/project/im2im/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/im2im)](https://pypi.org/project/im2im/)
[![Documentation](https://img.shields.io/badge/Doc-ReadMe-blue)](https://github.com/c3di/im2im/blob/main/README.md)
[![Tests](https://github.com/c3di/im2im/actions/workflows/python%20tests%20with%20coverage.yml/badge.svg)](https://github.com/c3di/im2im/actions/workflows/python%20tests%20with%20coverage.yml)
[![codecov](https://codecov.io/github/c3di/im2im/graph/badge.svg?token=BWBXANX8W7)](https://codecov.io/github/c3di/im2im)

The `im2im` package provides an automated approach for converting in-memory image representations across a variety of image processing libraries, including `numpy`, `opencv-python`, `torch`, `PIL`, and `Tensorflow`. It handles the nuances inherent to each library's image representation, such as data formats (numpy arrays, PIL images, torch tensors, and so on), color channel (RGB or grayscale), channel order (channel first or last or none), device (CPU/GPU), and pixel intensity ranges.


At the core of the package is a knowledge graph, where each node encapsulates metadata detailing an image representation, and the edges between nodes represent code snippets for converting images from one representation to another. When converting from the source to the target, the `im2im` package identifies the shortest path within the graph,  it gathers all relevant conversion snippets encountered along the path. These snippets are then combined to formulate the final conversion code, which is subsequently employed to transform the source images into the desired target format.




## Installation

Install the package via pip:
```bash
pip install im2im
```
## Usage

One example from the image data in numpy to the image data in PyTorch:
```python
from im2im import im2im

source_image_desc = {"lib": "numpy"}
target_image_desc = {"lib": "torch", "image_dtype": "uint8"}
code = im2im("source_image", source_image_desc, "target_image", target_image_desc)
```
The generated conversion code will be as follows:
```python
import torch
image = torch.from_numpy(source_image)
image = image.permute(2, 0, 1)
target_image = torch.unsqueeze(image, 0)
```



We use A* to find the shortest path in the knowledge graph. The goal function is only step cost by default. You can call `config_astar_goal_function` to set `cpu penalty`, `gpu penalty`  or enable `execution time cost` to the goal function. 

```python
...
config_astar_goal_function(cpu_penalty, gpu_penalty, include_time_cost, test_img_size=(256, 256))
```

## Evaluation


**Accuracy**

All primitive conversion code snippets are stored within the edges of the knowledge graph. These snippets are verified through execution checks to guarantee their correctness. For a more in-depth examination, please refer to the [`test_conversion_code_in_kg.py`](https://github.com/c3di/im2im/blob/main/tests/test_conversion_code_in_kg.py).

**Performance profiling**

The performance of knowledge graph construction and code generation processes is meticulously analyzed using the cProfile module. For comprehensive insights, please refer to the [`profiling notebooks`](https://github.com/c3di/im2im/blob/main/profile).

**Usability Evaluation**

Please refer to [Usability Evaluation](https://github.com/c3di/im2im_evaluation).

## Contribution
We welcome all contributions to this project! If you have suggestions, feature requests, or want to contribute in any other way, please feel free to open an issue or submit a pull request.


For detailed instructions on developing, building, and publishing this package, please refer to the [README_DEV](https://github.com/c3di/im2im/blob/main/README_Dev.md).


## Cite

if you use our tool or code in your research, please cite the following paper:

Todo

## License

This project is licensed under the MIT License. See the LICENSE file for details.
