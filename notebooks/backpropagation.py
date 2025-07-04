# %%
import numpy as np

# Passed gradient from the next layer
dvalues = np.array([1., 1., 1.])

# 3 sets of inputs, one for each neuron, we keep weights transposed
weights = np.array([
    [0.2, 0.8, -0.5, 1.],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T

# sum weights for each neuron and multiply with the gradient from the next layer
dx0 = sum(weights[0]) * dvalues[0]
dx1 = sum(weights[1]) * dvalues[0]
dx2 = sum(weights[2]) * dvalues[0]
dx3 = sum(weights[3]) * dvalues[0]

dinputs = np.array([dx0, dx1, dx2, dx3])
# dinputs is a gradient of the neuron function with respect to inputs
print(dinputs)
# %%
# Achieve the same result by doing the dot product
dinputs = np.dot(dvalues, weights.T)
print(dinputs)
# %%
# Account for a batch of samples and not just for single gradient vector that is propagated backwards between layers
dvalues = np.array([
    [1., 1., 1.],
    [2., 2., 2.],
    [3., 3., 3.]
])

dinputs = np.dot(dvalues, weights.T)
print(dinputs)
# %%
# Gradients are used to update the weights, for that gradients are supposed to be in the same shape as that of weights
# derivatives with respect to the weights equals dinputs * inputs
dvalues = np.array([
    [1., 1., 1.],
    [2., 2., 2.],
    [3., 3., 3.]
])

# Three sets of inputs or samples
inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

biases = np.array([2., 3., 0.5])

dweights = np.dot(inputs.T, dvalues)
dbiases = np.sum(dvalues, axis=0, keepdims=True) # Summing column wise
print(f"Gradients for weights: \n{dweights}\nGradient for biases: {dbiases}")
# %%
# For the backward pass, Activation function receives a gradient of the same shape as that of the layer's outputs
# Example layer output
z = np.array([
    [1., 2., -3., -4.],
    [2., -7., -1., 3.],
    [-1., 2., 5., -1.]
])

# Gradient from the next layer
dvalues = np.array([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10, 11, 12]
])

# derivative of ReLu
da_dz = np.zeros_like(z)
da_dz[z > 0] = 1
print(da_dz)
# The chain rule
da_dz = da_dz * dvalues
print(da_dz)
