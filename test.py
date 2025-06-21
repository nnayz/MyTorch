# %%
import torch
# import numpy as np

x = torch.Tensor([2.0, 3.0], requires_grad=True)
y = torch.Tensor([4.0, 5.0], requires_grad=True)

z = (x * y + x)
loss = z * torch.Tensor(2.0)
loss.backward()
# %%
print(f"Loss: {loss} | Grad: {x.grad}")
print(f"Computational graph: {loss._prev}")

print(x.grad)
print(y.grad)
