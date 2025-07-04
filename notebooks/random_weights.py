# %%
from layers import Layer_Dense
from activation_functions import ReLu, softmax
import numpy as np
import nnfs
from nnfs.datasets import vertical_data, spiral_data
# import matplotlib.pyplot as plt
from cost_functions import CCE
from metrics import accuracy

# %%
nnfs.init()
X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

# %%
# Create model
dense1 = Layer_Dense(n_neurons=3, n_inputs=2, activation_fn=ReLu)
dense2 = Layer_Dense(n_neurons=3, n_inputs=3, activation_fn=softmax)

# Helper variables
lowest_loss = 999999

# copy() ensures a full copy instead of a reference
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):

    # Generate a new set of weights for the iteration
    # dense1.weights = 0.05 * np.random.randn(2, 3)
    # dense1.biases = 0.05 * np.random.randn(1, 3)
    # dense2.weights = 0.05 * np.random.randn(3, 3)
    # dense2.biases = 0.05 * np.random.randn(1, 3)
    #
    # Instead of generating new weights, try to adjust weights using a fraction of the randomly generated numbers
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of the training data
    dense1.forward(X)
    dense2.forward(dense1.a)

    # Get loss using categorical cross entropy function
    loss = np.mean(CCE(y_true=y, y_pred=dense2.a))

    # Get accuracy
    accuracy_value = accuracy(y_pred=dense2.a, y_true=y)

    # If loss is smaller, print and save weights aside
    if loss < lowest_loss:
        print(f"New set of weights found, iteration: {iteration}\nloss: {loss}\nAccuracy: {accuracy_value}\n")
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    # If the loss comes out to be more than the previous best loss, we go back to previous weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

# %%
# Lets try this with the spiral dataset
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

dense1 = Layer_Dense(n_neurons=3, n_inputs=2, activation_fn=ReLu)
dense2 = Layer_Dense(n_neurons=3, n_inputs=3, activation_fn=softmax)

# Helper variables
lowest_loss = 999999

# copy() ensures a full copy instead of a reference
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):

    # Generate a new set of weights for the iteration
    # dense1.weights = 0.05 * np.random.randn(2, 3)
    # dense1.biases = 0.05 * np.random.randn(1, 3)
    # dense2.weights = 0.05 * np.random.randn(3, 3)
    # dense2.biases = 0.05 * np.random.randn(1, 3)
    #
    # Instead of generating new weights, try to adjust weights using a fraction of the randomly generated numbers
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of the training data
    dense1.forward(X)
    dense2.forward(dense1.a)

    # Get loss using categorical cross entropy function
    loss = np.mean(CCE(y_true=y, y_pred=dense2.a))

    # Get accuracy
    accuracy_value = accuracy(y_pred=dense2.a, y_true=y)

    # If loss is smaller, print and save weights aside
    if loss < lowest_loss:
        print(f"New set of weights found, iteration: {iteration}\nloss: {loss}\nAccuracy: {accuracy_value}\n")
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    # If the loss comes out to be more than the previous best loss, we go back to previous weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
