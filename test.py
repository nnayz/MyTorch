import torch
import numpy as np

x = np.arange(1, 5, 1)
y = np.arange(5, 1, -1)
tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)

# print(f"Newly created tensor: {tensor_x}")
# print(f"The Shape of the tensor: {tensor_x.shape}")
#
print(f"{tensor_x * tensor_y}")
