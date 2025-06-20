from cost_functions import CategoricalCrossEntropy
from activation_functions import Softmax
import numpy as np

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.loss = CategoricalCrossEntropy()
        self.activation = Softmax()

    # Forward pass
    def forward(self, inputs, y_true):
    # output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.a
        # Calculate and return loss
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
    # Number of samples
        samples = len(dvalues)
        # If labels are one hot encoded
        # turn them into discrete values
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalise gradient
        self.dinputs = self.dinputs / samples
