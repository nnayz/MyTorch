import numpy as np

class Softmax:
    # Forward pass
    def softmax(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalisation of the exp values
        confidences = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.a = confidences
        return self.a

class ReLu:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.a = np.maximum(0, inputs)
        return self.a

    # Backward pass
    def backward(self, dvalues):
    # dvalues denote the derivative from the next layer
    # Making a copy because we need to modify the original variables
        self.dinputs = dvalues

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
