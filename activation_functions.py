import numpy as np

class Softmax:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalisation of the exp values
        confidences = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.a = confidences
        return self.a
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If one hot encoded, turn to discrete values
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        # copy
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalise gradient
        self.dinputs = self.dinputs / samples

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
