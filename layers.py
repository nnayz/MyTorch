# %%
import numpy as np
from activation_functions import ReLu
# from activation_functions import softmax

class Layer_Dense:
    def __init__(self, n_neurons, n_inputs, activation_fn=ReLu):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.array(np.zeros((1, n_neurons)))
        self.activation_fn = activation_fn

    def forward(self, inputs):
        # Output before activation
        self.z = np.dot(inputs, self.weights) + self.biases

        # Output after activation
        self.a = self.activation_fn(self.z)

# Example usage
# layer = Layer_Dense(n_neurons=2, n_inputs=2, activation_fn=softmax)
# inputs = np.array([
#     [2, 1],
#     [3, 4]
# ])
# layer.forward(inputs)
# print(layer.z)
# print(layer.a)
