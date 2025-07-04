import numpy as np
from . import Tensor


def relu(x):
    """Functional ReLU activation"""
    return x.relu()


def sigmoid(x):
    """Functional sigmoid activation"""
    return x.sigmoid()


def tanh(x):
    """Functional tanh activation"""
    return x.tanh()


def softmax(x, dim=-1):
    """Functional softmax activation"""
    # Subtract max for numerical stability
    x_max = Tensor(np.max(x.data, axis=dim, keepdims=True))
    x_shifted = x - x_max
    
    # Compute exponentials
    exp_x = Tensor(np.exp(x_shifted.data))
    
    # Compute sum
    sum_exp = exp_x.sum(axis=dim, keepdims=True)
    
    # Return softmax
    return exp_x / sum_exp


def log_softmax(x, dim=-1):
    """Functional log softmax"""
    # Subtract max for numerical stability
    x_max = Tensor(np.max(x.data, axis=dim, keepdims=True))
    x_shifted = x - x_max
    
    # Compute log sum exp
    exp_x = Tensor(np.exp(x_shifted.data))
    log_sum_exp = Tensor(np.log(exp_x.sum(axis=dim, keepdims=True).data))
    
    # Return log softmax
    return x_shifted - log_sum_exp


def mse_loss(predictions, targets, reduction='mean'):
    """Functional mean squared error loss"""
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    diff = predictions - targets
    loss = diff * diff
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def cross_entropy(predictions, targets, reduction='mean'):
    """Functional cross entropy loss"""
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    # Apply log softmax
    log_probs = log_softmax(predictions, dim=-1)
    
    # Compute negative log likelihood
    if targets.data.ndim == 1:  # Class indices
        batch_size = targets.data.shape[0]
        nll = -Tensor(log_probs.data[np.arange(batch_size), targets.data.astype(int)])
    else:  # One-hot encoded
        nll = -(targets * log_probs).sum(axis=-1)
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:  # 'none'
        return nll


def binary_cross_entropy(predictions, targets, reduction='mean'):
    """Functional binary cross entropy loss"""
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    # Clip predictions for numerical stability
    epsilon = 1e-7
    preds_clipped = Tensor(np.clip(predictions.data, epsilon, 1 - epsilon))
    
    # BCE formula: -[y*log(p) + (1-y)*log(1-p)]
    loss = -(targets * Tensor(np.log(preds_clipped.data)) + 
             (Tensor(1.0) - targets) * Tensor(np.log(1.0 - preds_clipped.data)))
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def binary_cross_entropy_with_logits(logits, targets, reduction='mean'):
    """Functional binary cross entropy with logits"""
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    # Apply sigmoid to logits
    sigmoid_logits = logits.sigmoid()
    
    # Use BCE
    return binary_cross_entropy(sigmoid_logits, targets, reduction=reduction)


def dropout(x, p=0.5, training=True):
    """Functional dropout"""
    if not training or p == 0.0:
        return x
    
    if p == 1.0:
        return Tensor(np.zeros_like(x.data))
    
    # Create dropout mask
    mask = np.random.rand(*x.shape) > p
    
    # Apply dropout and scale
    scale = 1.0 / (1.0 - p)
    output = Tensor(x.data * mask * scale, requires_grad=x.requires_grad)
    output.prev = {x}
    
    def _backward():
        if x.requires_grad:
            x.grad = (x.grad if x.grad is not None else np.zeros_like(x.data)) + (mask * scale)
    
    output._backward = _backward
    return output


def linear(x, weight, bias=None):
    """Functional linear transformation"""
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return output


def conv1d(x, weight, bias=None, stride=1, padding=0):
    """Simplified 1D convolution (placeholder for future implementation)"""
    # This is a placeholder - full convolution implementation would be more complex
    raise NotImplementedError("conv1d not implemented yet")


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """Simplified 2D convolution (placeholder for future implementation)"""
    # This is a placeholder - full convolution implementation would be more complex
    raise NotImplementedError("conv2d not implemented yet")


def max_pool1d(x, kernel_size, stride=None, padding=0):
    """Simplified 1D max pooling (placeholder for future implementation)"""
    raise NotImplementedError("max_pool1d not implemented yet")


def max_pool2d(x, kernel_size, stride=None, padding=0):
    """Simplified 2D max pooling (placeholder for future implementation)"""
    raise NotImplementedError("max_pool2d not implemented yet")


def avg_pool1d(x, kernel_size, stride=None, padding=0):
    """Simplified 1D average pooling (placeholder for future implementation)"""
    raise NotImplementedError("avg_pool1d not implemented yet")


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    """Simplified 2D average pooling (placeholder for future implementation)"""
    raise NotImplementedError("avg_pool2d not implemented yet")


def normalize(x, p=2.0, dim=1, eps=1e-12):
    """Functional normalization"""
    if p == 2.0:
        # L2 normalization
        norm = Tensor(np.sqrt(np.sum(x.data ** 2, axis=dim, keepdims=True)) + eps)
        return x / norm
    else:
        # Lp normalization
        norm = Tensor(np.power(np.sum(np.abs(x.data) ** p, axis=dim, keepdims=True), 1.0/p) + eps)
        return x / norm


def pad(x, pad_width, mode='constant', value=0):
    """Functional padding"""
    if mode == 'constant':
        padded_data = np.pad(x.data, pad_width, mode='constant', constant_values=value)
    elif mode == 'reflect':
        padded_data = np.pad(x.data, pad_width, mode='reflect')
    elif mode == 'replicate':
        padded_data = np.pad(x.data, pad_width, mode='edge')
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    
    output = Tensor(padded_data, requires_grad=x.requires_grad)
    output.prev = {x}
    
    def _backward():
        if x.requires_grad:
            # Remove padding from gradient
            grad = np.ones_like(output.data)
            
            # Create slices to remove padding
            slices = []
            for i, (pad_before, pad_after) in enumerate(pad_width):
                start = pad_before
                end = grad.shape[i] - pad_after if pad_after > 0 else grad.shape[i]
                slices.append(slice(start, end))
            
            unpadded_grad = grad[tuple(slices)]
            x.grad = (x.grad if x.grad is not None else np.zeros_like(x.data)) + unpadded_grad
    
    output._backward = _backward
    return output


# Alias for commonly used functions
F = type('F', (), {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softmax': softmax,
    'log_softmax': log_softmax,
    'mse_loss': mse_loss,
    'cross_entropy': cross_entropy,
    'binary_cross_entropy': binary_cross_entropy,
    'binary_cross_entropy_with_logits': binary_cross_entropy_with_logits,
    'dropout': dropout,
    'linear': linear,
    'normalize': normalize,
    'pad': pad,
})() 