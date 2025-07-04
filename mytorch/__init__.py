import numpy as np

class Tensor:
    def __init__(self, x, dtype=np.float32, requires_grad=False):
        self.data = np.asarray(x, dtype=dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad

        # Gradient calculation
        self.grad = None
        self._backward = lambda : None
        self.prev = set() # Parent tensors in a backwards order

    # Arithmetic operations
    def __add__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data + operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                # Handle broadcasting for gradients
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                # Sum over broadcasted dimensions
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = np.ones_like(operand.data)
                if output.grad is not None:
                    grad = output.grad
                # Sum over broadcasted dimensions
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __radd__(self, operand):
        return self.__add__(operand)

    def __sub__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data - operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = np.ones_like(operand.data)
                if output.grad is not None:
                    grad = output.grad
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) - grad
        output._backward = _backward

        return output

    def __rsub__(self, operand):
        return Tensor(operand).__sub__(self)

    def __mul__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data * operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = operand.data
                if output.grad is not None:
                    grad = output.grad * operand.data
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = self.data
                if output.grad is not None:
                    grad = output.grad * self.data
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __rmul__(self, operand):
        return self.__mul__(operand)

    def __truediv__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data / operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = (1.0 / operand.data)
                if output.grad is not None:
                    grad = output.grad * (1.0 / operand.data)
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = (-self.data / (operand.data ** 2))
                if output.grad is not None:
                    grad = output.grad * (-self.data / (operand.data ** 2))
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __rtruediv__(self, operand):
        return Tensor(operand).__truediv__(self)

    def __pow__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data ** operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = (operand.data * (self.data ** (operand.data - 1)))
                if output.grad is not None:
                    grad = output.grad * (operand.data * (self.data ** (operand.data - 1)))
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = (output.data * np.log(self.data))
                if output.grad is not None:
                    grad = output.grad * (output.data * np.log(self.data))
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __matmul__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        try:
            output = Tensor(self.data @ operand.data, requires_grad=(
                self.requires_grad or operand.requires_grad
            ))
            output.prev = {self, operand}

            def _backward():
                if self.requires_grad:
                    grad = np.ones_like(self.data)
                    if output.grad is not None:
                        grad = output.grad @ operand.data.T
                    else:
                        grad = np.ones_like(output.data) @ operand.data.T
                    self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
                if operand.requires_grad:
                    grad = np.ones_like(operand.data)
                    if output.grad is not None:
                        grad = self.data.T @ output.grad
                    else:
                        grad = self.data.T @ np.ones_like(output.data)
                    operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
            output._backward = _backward

            return output
        except Exception as e:
            raise RuntimeError(f"Matrix Multiplication not possible : {e}")

    def __neg__(self):
        output = Tensor(-self.data, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = -np.ones_like(self.data)
                if output.grad is not None:
                    grad = -output.grad
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    # Activation functions
    def relu(self):
        output = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(self.dtype)
                if output.grad is not None:
                    grad = output.grad * (self.data > 0).astype(self.dtype)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        output = Tensor(sig, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = sig * (1 - sig)
                if output.grad is not None:
                    grad = output.grad * sig * (1 - sig)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def tanh(self):
        tanh_val = np.tanh(self.data)
        output = Tensor(tanh_val, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = (1 - tanh_val ** 2)
                if output.grad is not None:
                    grad = output.grad * (1 - tanh_val ** 2)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    # Utility functions
    def sum(self, axis=None, keepdims=False):
        output = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                    # Broadcast gradient back to original shape
                    if axis is not None:
                        if not keepdims:
                            # Add back the reduced dimensions
                            if isinstance(axis, int):
                                grad = np.expand_dims(grad, axis=axis)
                            else:
                                for ax in sorted(axis):
                                    grad = np.expand_dims(grad, axis=ax)
                        # Broadcast to original shape
                        grad = np.broadcast_to(grad, self.data.shape)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def mean(self, axis=None, keepdims=False):
        output = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                if axis is None:
                    grad = np.ones_like(self.data) / self.data.size
                else:
                    axis_size = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[ax] for ax in axis])
                    grad = np.ones_like(self.data) / axis_size
                
                if output.grad is not None:
                    grad = output.grad
                    # Broadcast gradient back to original shape
                    if axis is not None:
                        if not keepdims:
                            # Add back the reduced dimensions
                            if isinstance(axis, int):
                                grad = np.expand_dims(grad, axis=axis)
                            else:
                                for ax in sorted(axis):
                                    grad = np.expand_dims(grad, axis=ax)
                        # Broadcast to original shape
                        grad = np.broadcast_to(grad, self.data.shape)
                        if axis is None:
                            grad = grad / self.data.size
                        else:
                            axis_size = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[ax] for ax in axis])
                            grad = grad / axis_size
                
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def reshape(self, shape):
        output = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(output.data)
                if output.grad is not None:
                    grad = output.grad.reshape(self.data.shape)
                else:
                    grad = np.ones_like(output.data).reshape(self.data.shape)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def transpose(self, axes=None):
        output = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(output.data)
                if output.grad is not None:
                    grad = output.grad
                # Reverse the transpose operation
                if axes is None:
                    grad = np.transpose(grad)
                else:
                    back_axes = np.argsort(axes)
                    grad = np.transpose(grad, back_axes)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def backward(self):
        # Topological sort for backward pass
        visited = set()
        topo_order = []

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            for child in node.prev:
                build_topo(child)
            topo_order.append(node)

        build_topo(self)

        # Initialize gradient for the root node
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Backward pass in reverse topological order
        for node in reversed(topo_order):
            node._backward()

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"<mytorch.Tensor({self.data}){', requires_grad=True' if self.requires_grad else ''}>"


def tensor(x, dtype=np.float32, requires_grad=False):
    """Create a tensor similar to torch.tensor()"""
    return Tensor(x, dtype=dtype, requires_grad=requires_grad)


def zeros(shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with zeros"""
    return Tensor(np.zeros(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with ones"""
    return Tensor(np.ones(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def randn(*shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with random normal values"""
    return Tensor(np.random.randn(*shape).astype(dtype), dtype=dtype, requires_grad=requires_grad)


def rand(*shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with random uniform values"""
    return Tensor(np.random.rand(*shape).astype(dtype), dtype=dtype, requires_grad=requires_grad)
