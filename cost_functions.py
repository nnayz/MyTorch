import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error and returns it
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if y_true.ndim != y_pred.ndim:
        raise ValueError(
            f"The dimensions of true values and the predicted values are not the same\nDimensions y_true: {y_true.ndim} \nDimensions y_pred: {y_pred.ndim} "
        )
    errors = y_true - y_pred
    errors_squared = errors ** 2
    mean_squared_error = np.mean(errors_squared)
    return mean_squared_error
