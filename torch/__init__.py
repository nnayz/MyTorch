import numpy as np

class Tensor:
    def __init__(self, x, dtype=np.float32, requires_grad=False):
        self.data = np.asarray(x, dtype=dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = None # Will store the gradient computed
        self._prev = set() # Parent tensors
        self._backward = lambda : None

    @property
    def T(self):
        output = Tensor(self.data.T, requires_grad=self.requires_grad, dtype=self.dtype)
        return output

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"torch.Tensor({self.data}), <requires_grad={self.requires_grad}>"

    # Elementary arithmetic operations
    def __add__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        output = Tensor(self.data + operand.data,
            requires_grad=(self.requires_grad or operand.requires_grad
        ))
        output._prev = {self, operand}

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + np.ones_like(self.data)
            if operand.requires_grad:
                operand.grad = (operand.grad or np.zeros_like(operand.data)) + np.ones_like(operand.data)

        output._backward = _backward
        return output

    def __mul__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        output = Tensor(self.data * operand.data,
            requires_grad=(self.requires_grad or operand.requires_grad
        ))
        output._prev = {self, operand}

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + operand.data
            if operand.requires_grad:
                operand.grad = (operand.grad or np.zeros_like(operand.data)) + self.data

        output._backward = _backward
        return output

    def __matmul__(self, operand):
        if not isinstance(operand, Tensor):
            raise TypeError("Operand must be a Tensor")


        try:
            output = Tensor(self.data @ operand.data, requires_grad=(self.requires_grad or operand.requires_grad))
            output._prev = {self, operand}
        except Exception as e:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} @ {operand.shape}") from e

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + operand.data
            if operand.requires_grad:
                operand.grad = (operand.grad or np.zeros_like(operand.data)) + self.data
        output._backward = _backward
        return output


    # Backpropagation
    def backward(self):
        if self.ndim != 0:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")

        if not self.requires_grad:
            raise RuntimeError("Called backward on a non leaf tensor with requires_grad=False")

        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data) # d_output / d_output = 1

        for node in reversed(topo):
            node._backward()
