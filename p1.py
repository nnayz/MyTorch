import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

np.random.seed(0)

X = [[   1,   2,   3,  2.5],
     [   2,   5,  -1,    2],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5) # 5 number of neurons
# layer2 = Layer_Dense(5, 2)
activation1 = Activation_ReLu() # activation function
layer1.forward(X)
# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

# layer2.forward(layer1.output)
# print(layer2.output)

