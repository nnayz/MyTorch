import numpy as np
from . import Tensor, zeros, ones, randn


class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def parameters(self):
        """Return all parameters in this module and its submodules"""
        params = []
        for param in self._parameters.values():
            if isinstance(param, Tensor):
                params.append(param)
        
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
        
        return params
    
    def named_parameters(self):
        """Return all named parameters in this module and its submodules"""
        params = []
        for name, param in self._parameters.items():
            if isinstance(param, Tensor):
                params.append((name, param))
        
        for module_name, module in self._modules.items():
            if isinstance(module, Module):
                for name, param in module.named_parameters():
                    params.append((f"{module_name}.{name}", param))
        
        return params
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
    
    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Linear(Module):
    """Linear transformation layer: y = xW^T + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights with Xavier/Glorot initialization
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        # x shape: (batch_size, in_features)
        # weight shape: (out_features, in_features)
        # output shape: (batch_size, out_features)
        
        output = x @ self.weight.T  # Matrix multiplication
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *args):
        super().__init__()
        self.layers = []
        for i, module in enumerate(args):
            self.layers.append(module)
            self._modules[str(i)] = module
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def add_module(self, module):
        idx = len(self.layers)
        self.layers.append(module)
        self._modules[str(idx)] = module


# Activation function modules
class ReLU(Module):
    """ReLU activation function"""
    
    def forward(self, x):
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def forward(self, x):
        return x.sigmoid()
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation function"""
    
    def forward(self, x):
        return x.tanh()
    
    def __repr__(self):
        return "Tanh()"


# Loss functions
class MSELoss(Module):
    """Mean Squared Error Loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        diff = predictions - targets
        loss = (diff * diff).mean()
        return loss
    
    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """Cross Entropy Loss (simplified version)"""
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        # Softmax
        exp_preds = Tensor(np.exp(predictions.data - np.max(predictions.data, axis=1, keepdims=True)))
        softmax = exp_preds / exp_preds.sum(axis=1, keepdims=True)
        
        # Cross entropy
        # For numerical stability, clip values
        epsilon = 1e-7
        softmax_data = np.clip(softmax.data, epsilon, 1 - epsilon)
        
        if targets.data.ndim == 1:  # Class indices
            batch_size = targets.data.shape[0]
            log_likelihood = -np.log(softmax_data[np.arange(batch_size), targets.data.astype(int)])
        else:  # One-hot encoded
            log_likelihood = -np.sum(targets.data * np.log(softmax_data), axis=1)
        
        loss = Tensor(np.mean(log_likelihood), requires_grad=True)
        
        # Manual backward pass for cross entropy
        def _backward():
            if predictions.requires_grad:
                if targets.data.ndim == 1:
                    grad = softmax.data.copy()
                    grad[np.arange(len(targets.data)), targets.data.astype(int)] -= 1
                else:
                    grad = softmax.data - targets.data
                grad = grad / len(targets.data)
                predictions.grad = (predictions.grad if predictions.grad is not None else np.zeros_like(predictions.data)) + grad
        
        loss._backward = _backward
        loss.prev = {predictions}
        
        return loss
    
    def __repr__(self):
        return "CrossEntropyLoss()"


class BCELoss(Module):
    """Binary Cross Entropy Loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        # Clip predictions for numerical stability
        epsilon = 1e-7
        preds_clipped = Tensor(np.clip(predictions.data, epsilon, 1 - epsilon))
        
        # BCE formula: -[y*log(p) + (1-y)*log(1-p)]
        loss = -(targets * Tensor(np.log(preds_clipped.data)) + 
                (Tensor(1.0) - targets) * Tensor(np.log(1.0 - preds_clipped.data)))
        
        return loss.mean()
    
    def __repr__(self):
        return "BCELoss()"


# Utility functions
def init_weights(module, init_type='xavier'):
    """Initialize weights of a module"""
    if isinstance(module, Linear):
        if init_type == 'xavier':
            bound = np.sqrt(6.0 / (module.in_features + module.out_features))
            module.weight.data = np.random.uniform(-bound, bound, module.weight.shape).astype(np.float32)
        elif init_type == 'normal':
            module.weight.data = np.random.randn(*module.weight.shape).astype(np.float32) * 0.01
        elif init_type == 'zero':
            module.weight.data = np.zeros(module.weight.shape, dtype=np.float32)
        
        if module.bias is not None:
            module.bias.data = np.zeros(module.bias.shape, dtype=np.float32)
