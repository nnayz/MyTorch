import torch

x = torch.Tensor([2., 3.], requires_grad=True)
y = torch.Tensor([4., 5.], requires_grad=True)

z = x + y
loss = z * 2 @ z
loss.backward()

print(z.grad)
print(x.grad)
