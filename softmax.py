import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]
# e = 2.718281828459045
exp_values = np.exp(layer_outputs)
print(exp_values)

# norm_base = np.sum(exp_values)
# norm_values = exp_values / norm_base

# print(norm_values)

# print(sum(norm_values))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
