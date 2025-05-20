import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function. It squashes any input value in the range of 0 and 1
    """
    return 1 / (1 + np.exp(-x))

def ll(x):
    return x

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs) # Each input will have a weight associated with it
        self.bias = np.random.rand(1)[0] # Single bias value

        print(f"Neuron created with {n_inputs} inputs")

    def forward(self, inputs):
        """
        Neuron processes the inputs
        """
        # Calculate weighted sum by dot product of inputs and weights
        weighted_sum = np.dot(inputs, self.weights)
        output_before_activation = weighted_sum + self.bias

        # Apply the activation function
        output = sigmoid(output_before_activation)
        return output


# Create a neuron that has two inputs
neuron = Neuron(2)

inputs = np.array([0.5, -1.0])
output = neuron.forward(inputs)
print(output)
