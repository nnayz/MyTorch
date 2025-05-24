# %%
# Using chain rule to decrease the neuron's output
import numpy as np
from activation_functions import ReLu

x = np.array([1.0, -2.0, 3.0]) # inputs
w = np.array([-3.0, -1.0, 2.0]) # Weights
fn = ReLu()

b = 1.0

# doing it step by step
# Multiplying inputs with weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted sum with bias ( output before activation )
z = xw0 + xw1 + xw2 + b

a = fn.forward(z)
print(f"Output before activation : {z}\nOutput after activation : {a}")

# Partial derivatves of z with respect to each of the weighted sums and bias
dz_dxw0 = 1.0
dz_dxw1 = 1.0
dz_dxw2 = 1.0
dz_db = 1.0

# Lets assume the derivative from the next layer is 1.0
dvalue = 1.

def derivative_ReLu(z):
    return (1. if z > 0 else 0.)

da_dz = dvalue * derivative_ReLu(z)

da_dxw0 = da_dz * dz_dxw0
da_dxw1 = da_dz * dz_dxw1
da_dxw2 = da_dz * dz_dxw2
da_db = da_dz * dz_db
print(da_dxw0, da_dxw1, da_dxw2, da_db)
# Partial derivatives of z with respect to each of the inputs
dz_dx0 = w[0]
dz_dx1 = w[1]
dz_dx2 = w[2]

# Partial derivatives of z with respect to each of the weights
dz_dw0 = x[0]
dz_dw1 = x[1]
dz_dw2 = x[2]

# Partial derivatives of activation function with respect to each of the weights and inputs
da_dx0 = da_dz * dz_dx0
da_dx1 = da_dz * dz_dx1
da_dx2 = da_dz * dz_dx2
da_dw0 = da_dz * dz_dw0
da_dw1 = da_dz * dz_dw1
da_dw2 = da_dz * dz_dw2

print(da_dx0, da_dw0, da_dx1, da_dw1, da_dx2, da_dw2)

# %%
# Gradient on inputs
dx = np.array([da_dx0, da_dx1, da_dx2])

# Gradient on weights
dw = np.array([da_dw0, da_dw1, da_dw2])

# Gradient on bias
db = da_db
print(f"Gradient on inputs: \n{dx}\n")
print(f"Gradient on weights: \n{dw}\n")
print(f"Gradient on bias: \n{db}\n")
print(f"Weights before applying: \n{w}\n")
# Apply a fraction of the gradients to the current weights
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(f"Weights after applying: \n{w}\n")
# %%
# Perform a forward pass again
# Multiplying inputs with weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted sum with bias ( output before activation )
z = xw0 + xw1 + xw2 + b

a = fn.forward(z)
print(f"Output before activation : {z}\nOutput after activation : {a}")
# Successfully decreased this neuron’s output from 6.000 to 5.985.
