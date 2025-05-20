import numpy as np

def mean_squared_error(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=float)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shapes of y_pred and y_true must be same \n Shape of y_true {y_true.shape}, Shape pf y_pred {y_pred.shape}"
        )

    errors = y_true - y_pred
    error_squared = errors ** 2
    mean_squared_error = error_squared / np.sum(error_squared)
    return mean_squared_error
