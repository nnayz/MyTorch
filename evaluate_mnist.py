import numpy as np
import pandas as pd
import mytorch.nn as nn
from mytorch import tensor
from collections import defaultdict
from tqdm import tqdm


def load_data(test_path):
    data = pd.read_csv(test_path)
    X = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = data.iloc[:, 0].values.astype(np.int32)
    return X, y


def create_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def load_weights(model, weight_file='mnist_weights.npz'):
    weights = np.load(weight_file)
    for name, param in model.named_parameters():
        if name in weights:
            param.data = weights[name]
        else:
            raise KeyError(f"Weight for parameter '{name}' not found in file.")
    print("Weights loaded successfully.")


def evaluate(model, X, y, batch_size=128):
    model.eval()
    preds = []
    true = []
    for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        outputs = model(tensor(batch_X))
        predictions = np.argmax(outputs.data, axis=1)
        preds.append(predictions)
        true.append(batch_y)
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    return preds, true


def compute_metrics(preds, true, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true, preds):
        cm[t, p] += 1
    accuracy = np.trace(cm) / np.sum(cm)
    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)
        if prec + rec == 0:
            f1.append(0.0)
        else:
            f1.append(2 * prec * rec / (prec + rec))
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': np.mean(precision),
        'recall_macro': np.mean(recall),
        'f1_macro': np.mean(f1),
        'confusion_matrix': cm,
    }
    return metrics


def main():
    X_test, y_test = load_data('archive/mnist_test.csv')
    model = create_model()
    load_weights(model, 'mnist_weights.npz')
    preds, true = evaluate(model, X_test, y_test, batch_size=256)
    metrics = compute_metrics(preds, true)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score: {metrics['f1_macro']:.4f}\n")

    print("Per-class metrics:")
    for c in range(10):
        print(f"Class {c}: Precision {metrics['precision_per_class'][c]:.4f}, "
              f"Recall {metrics['recall_per_class'][c]:.4f}, "
              f"F1 {metrics['f1_per_class'][c]:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


if __name__ == '__main__':
    main() 