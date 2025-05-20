# %%
import numpy as np

# %%
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# %%
class Dense_Layer:
    def __init__(self, n_neurons, n_inputs):
        """
        n_inputs -> number of inputs per neuron
        n_neurons -> number of neurons
        """
        self.weights = np.random.rand(n_neurons, n_inputs)
        self.biases = np.random.rand(n_neurons)

        print(f"Layer created with {n_neurons} neurons, each expecting {n_inputs} inputs")

    def forward(self, inputs):
        """
        Inputs -> 1D array or a vector of all the input values
        """
        print(self.weights)
        print(inputs)
        weighted_sum = np.dot(self.weights, inputs)
        output_before_activation = weighted_sum + self.biases
        output = sigmoid(output_before_activation)
        return output

# %%
layer = Dense_Layer(3, 2) # 3 neurons with 2 inputs to each neuron
example_inputs = np.array([0.5, 1.0]) # 2 input values
output = layer.forward(example_inputs)

print(f"Output: {output} with shape {output.shape}")

# %%
class Input_Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons # Represents the number of features

    def receive_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.ndarray(data)

        if data.shape[0] != self.n_neurons and data.ndim != 1:
            raise ValueError(
                f"Expected a 1D Array with {self.n_neurons} features but received an array of {data.shape} dimensions"
            )

        return data

# %%
n_features = 2 # For instance, height and weight
data = np.array([160.0, 55.0]) # Representing height and weight
input_layer = Input_Layer(2)
input_layer.receive_data(data)
