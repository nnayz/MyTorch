# This file is intentionally left blank
import numpy as np

class Tensor:
    def __init__(self, x: np.ndarray, requires_grad=False):
        self.data = np.asarray(x)
        self.requires_grad = requires_grad
        self.grad = None # Will store the gradient computed
        self._prev = set() # Parent tensors

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"torch.Tensor({self.data}), \n<requires_grad={self.requires_grad}>"

    # Elementary arithmetic operations
    def __add__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        output = Tensor(self.data + operand.data)
        output._prev = {self, operand}

        return output

    def __mul__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        output = Tensor(self.data * operand.data)
        output._prev = {self, operand}

        return output
