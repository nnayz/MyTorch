import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0] # One hot encoding
target_class = 0

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

# print the loss for the actual class
loss = -math.log(softmax_output[target_class])

print(loss)


