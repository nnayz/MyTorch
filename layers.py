# %%
import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, ReLu_derivative, ReLu, linear_activation, linear_activation_derivative

# %%
class Input_Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons # Represents the number of features
        print(f"Input Layer created with n_neurons or features as {n_neurons}")

    def receive_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.ndarray(data, dtype=float)

        if data.shape[0] != self.n_neurons and data.ndim != 1:
            raise ValueError(
                f"Expected a 1D Array with {self.n_neurons} features but received an array of {data.ndim} dimensions"
            )

        return data

# %%
class Dense_Layer:
    def __init__(self, n_neurons, n_inputs, activation_fn=sigmoid):
        """
        n_inputs -> number of inputs per neuron
        n_neurons -> number of neurons
        """
        self.weights = np.random.rand(n_neurons, n_inputs)
        self.biases = np.random.rand(n_neurons)
        self.activation_fn = activation_fn

        if activation_fn == sigmoid:
            self.activation_fn_derivative = sigmoid_derivative
        elif activation_fn == ReLu:
            self.activation_fn_derivative = ReLu_derivative
        elif activation_fn == linear_activation:
            self.activation_fn_derivative = linear_activation_derivative

        # Placeholders for gradients (same shape as weights and biases)
        self.dweights = np.zeros_like(self.weights) # In other words, how much each weight contributed to the error
        self.dbiases = np.zeros_like(self.biases) # How much each bias contributed to the error

        # To store the input during forward pass
        # self.inputs_cache = None

        # To store the output before activation
        self.z_cache = None

        # To store the output after activation
        self.activation_cache = None

        print(f"Layer created with {n_neurons} neurons, each expecting {n_inputs} inputs")

    def forward(self, inputs):
        """
        Inputs -> 1D array or a vector of all the input values
        """
        self.inputs_cache = inputs # Store inputs
        weighted_sum = np.dot(self.weights, inputs)
        self.z_cache = weighted_sum + self.biases
        self.activation_cache = self.activation_fn(self.z_cache)
        return self.activation_cache
