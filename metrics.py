import numpy as np

def accuracy(y_true, y_pred):
    class_predictions = np.argmax(y_pred, axis=1)
    if y_true.ndim == 1:
        accuracy = np.mean(class_predictions == y_true)
    else:
        y_true = np.argmax(y_true, axis=1)
        accuracy = np.mean(class_predictions == y_true)

    return accuracy

# softmax_outputs = np.array([
#     [0.8, 0.1, 0.1],
#     [0.3, 0.6, 0.1],
#     [0.2, 0.6, 0.2]
# ])

# # One hot encoded
# correct_classes = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0]
# ])

# # Class targets
# class_targets = np.array([0, 1, 1])

# print(accuracy(class_targets, softmax_outputs))
# print(accuracy(correct_classes, softmax_outputs))
