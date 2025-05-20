import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    """
      Calculates Categorical Cross-Entropy (CCE) loss.

      Args:
        y_true: A 2D NumPy array of true target values (one-hot encoded).
                Shape: (n_samples, n_classes)
        y_pred: A 2D NumPy array of predicted probabilities from the network's
                softmax output. Shape: (n_samples, n_classes)

      Returns:
        A single float value representing the mean CCE loss.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=float)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shapes of y_pred and y_true must be same \n Shape of y_true {y_true.shape}, Shape pf y_pred {y_pred.shape}"
        )

    if y_true.ndim != 2 or y_pred.ndim != 2:
          raise ValueError("Inputs y_true and y_pred must be 2D arrays (batch_size, num_classes).")

    # Clipping to avoid log(0) and log(1)
    epsilon = 1e-12
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)

    correct_confidences = np.sum(y_true * y_pred_clipped, axis=1)
    negative_log_likelihoods = -np.log(correct_confidences)
    mean_loss = np.mean(negative_log_likelihoods)
    return mean_loss

y_true_example = np.array([
    [0, 1, 0],  # True class is 1
    [1, 0, 0],  # True class is 0
    [0, 0, 1]   # True class is 2
])

# Predictions from a softmax layer
y_pred_example_good = np.array([
    [0.1, 0.8, 0.1],  # Predicts class 1 with 80% (good)
    [0.7, 0.2, 0.1],  # Predicts class 0 with 70% (good)
    [0.2, 0.1, 0.7]   # Predicts class 2 with 70% (good)
])

y_pred_example_bad = np.array([
    [0.7, 0.1, 0.2],  # Predicts class 0 with 70% (bad, true is 1)
    [0.1, 0.8, 0.1],  # Predicts class 1 with 80% (bad, true is 0)
    [0.6, 0.2, 0.2]   # Predicts class 0 with 60% (bad, true is 2)
])

loss_good = categorical_cross_entropy(y_true_example, y_pred_example_good)
loss_bad = categorical_cross_entropy(y_true_example, y_pred_example_bad)

print(f"y_true:\n{y_true_example}")
print(f"y_pred (good predictions):\n{y_pred_example_good}")
print(f"CCE Loss (good predictions): {loss_good:.4f}") # Should be relatively low

print("-" * 30)

print(f"y_pred (bad predictions):\n{y_pred_example_bad}")
print(f"CCE Loss (bad predictions): {loss_bad:.4f}") # Should be higher
