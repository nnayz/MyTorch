import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, y_true, y_pred):
        # Calculate sample loss
        sample_losses = self.forward(y_true, y_pred)

        # Ensure the result is a NumPy array before taking the mean
        if sample_losses is None:
            raise ValueError("Loss forward() returned None. Check implementation.")


        # Calculate mean
        return np.mean(sample_losses)

    @abstractmethod
    def forward(self, y_true, y_pred):
        pass


class CategoricalCrossEntropy(Loss):
    def forward(self, y_true, y_pred):
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
    def backward(self, dvalues, y_true):
    # Number of samples
        samples = len(dvalues)

        # Number of labels
        labels = len(dvalues[0])

        # Convert to one hot
        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples
