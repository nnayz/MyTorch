import numpy as np

# Sigmoid activation function
def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(output_of_sigmoid):
    """
    sigmoid(x) = s, sigmoid_derivative = s * (1 - s)
    """
    return output_of_sigmoid * (1 - output_of_sigmoid)

# Softmax activation function
def softmax(x):
    """
    Args:
        1D, 2D array of raw scores.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    if x.ndim == 2:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise ValueError(f"The input should be a 1D or 2D array, instead got {x.ndim}D array")

def ReLu_derivative(x_input_to_relu):
    """Derivative of ReLU w.r.t. its input x."""
    # Derivative is 1 if x > 0, else 0
    return np.where(x_input_to_relu > 0, 1, 0)

def linear_activation_derivative(x_input_to_linear):
    """Derivative of linear activation w.r.t. its input x."""
    # If f(x) = x, then f'(x) = 1.
    # The argument isn't strictly needed but kept for consistency if dispatching.
    # We need an array of 1s of the same shape as x_input_to_linear.
    return np.ones_like(x_input_to_linear)

# ReLu -> Rectified Linear Unit activation function
def ReLu(x):
    """
    Args:
        1D array or scalar
    """
    return np.maximum(0, x)

# Linear activation (no activation f(x) = x)
def linear_activation(x):
    return x
