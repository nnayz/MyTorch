# %%
import numpy as np

# %%
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax activation function
def softmax(x):
    """
    Args:
        1D array of raw scores.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise ValueError(f"The input should be a 1D array, instead got {x.ndim}")

# ReLu -> Rectified Linear Unit activation function
def ReLu(x):
    return 0 if x < 0 else x

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

        print(f"Layer created with {n_neurons} neurons, each expecting {n_inputs} inputs")

    def forward(self, inputs):
        """
        Inputs -> 1D array or a vector of all the input values
        """
        print(self.weights)
        print(inputs)
        weighted_sum = np.dot(self.weights, inputs)
        output_before_activation = weighted_sum + self.biases
        if self.activation_fn == softmax:
            output = softmax(output_before_activation)
            return output
        elif self.activation_fn == ReLu:
            output = ReLu(output_before_activation)
            return output
        output = sigmoid(output_before_activation)
        return output


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
# Neural Network
input_layer = Input_Layer(2)
hidden_layer = Dense_Layer(n_neurons=3, n_inputs=2)
hidden_layer2 = Dense_Layer(n_neurons=3, n_inputs=3)
output_layer = Dense_Layer(n_neurons=2, n_inputs=3, activation_fn=softmax)
