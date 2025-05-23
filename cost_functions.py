import numpy as np

def CCE(y_true, y_pred):
    # Number of samples in a batch
    samples = len(y_pred)

    # Clipping y_pred to avoid dealing with log(0)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Probabilities for target values
    # Only if categorical labels
    # For eg. true_classes = [0, 1, 2]
    if y_true.ndim == 1:
        correct_confidences = y_pred_clipped[
            range(samples),
            y_true
        ]

    # Mask values
    # For one-hot encoded y_true
    # i.e y_true.ndim == 2
    else:
        correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
        )

    # loss
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods


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

# print(CCE(correct_classes, softmax_outputs)) # One hot encoded
# print(CCE(class_targets, softmax_outputs)) # Class targets, Categorical Labels
