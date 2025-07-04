# %%
import numpy as np
# from activation_functions import softmax, ReLu

class Layer_Dense:
    def __init__(self, n_neurons, n_inputs):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.array(np.zeros((1, n_neurons)))

    def forward(self, inputs):
        # Output before activation
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
    # The original variables will be modified, make a copy
        self.dinputs = dvalues.copy()

        # Zero gradients where input values were negative
        self.dinputs[dvalues <= 0] = 0
