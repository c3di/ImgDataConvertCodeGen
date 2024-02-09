from datetime import datetime
import numpy as np
import torch

from measure import get_execution_time
from src.imgdataconvertcodegen import get_covert_code, get_convert_path

print('-' * 10, 'Task: Get Conversion Code with metadata as input', '-' * 10)
begin_time = datetime.now()
source_metadata = {
    "data_representation": "numpy.ndarray",
    "color_channel": "rgb",
    "channel_order": "channel last",
    "minibatch_input": False,
    "data_type": "uint8",
    "intensity_range": "0to255",
    "device": "cpu"
}
target_metadata = {
    "data_representation": "torch.tensor",
    "color_channel": "rgb",
    "channel_order": "channel first",
    "minibatch_input": True,
    "data_type": "uint8",
    "intensity_range": "0to255",
    "device": "cpu"
}

code = get_covert_code("input", source_metadata, "output", target_metadata)
end_time = datetime.now()
print(f"the source metadata is: \n\t{source_metadata}")
print(f"the target metadata is: \n\t{target_metadata}")
print(f"The conversion code is: \n\{code}")
print(get_execution_time(begin_time, end_time))
print('-' * 10, 'Task: Get Conversion Code with library name as input', '-' * 10)
code_using_lib_name = get_covert_code("input", 'scikit-image', "output", 'torch')
print(f"The conversion code is: \n{code_using_lib_name}")


print('-' * 10, 'Task: Get Conversion Path', '-' * 10)
begin_time = datetime.now()
path = get_convert_path(source_metadata, target_metadata)
end_time = datetime.now()
print("the path is:")
for metadata in path:
    print(f'\t{metadata}')
print(get_execution_time(begin_time, end_time))


print('-' * 10, 'Task: Convert Image Data', '-' * 10)
source_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
expected_image = torch.from_numpy(source_image).unsqueeze(0).permute(0, 3, 1, 2)
begin_time = datetime.now()
code = get_covert_code("source_image", source_metadata, "actual_image", target_metadata)
exec(code)
end_time = datetime.now()
if torch.equal(actual_image, expected_image):
    print(f'Success: expected and actual images are equal')
else:
    print(f'Error: expected and actual images are not equal')
print(get_execution_time(begin_time, end_time))