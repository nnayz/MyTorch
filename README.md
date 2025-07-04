# MyTorch

A PyTorch-like deep learning framework implementation for educational purposes. This project implements a complete neural network library with automatic differentiation, common layer types, optimizers, and example implementations.

## Features

- Tensor operations with automatic differentiation
- Neural network modules (Linear, ReLU, etc.)
- Optimizers (SGD, Adam)
- MNIST classification example with 97.16% accuracy
- Comprehensive evaluation metrics

## Setup

1. Clone the repository:
```bash
git clone https://github.com/nnayz/MyTorch.git
cd MyTorch
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download MNIST Dataset:
The MNIST dataset files are not included in the repository due to their size. You'll need to:

a. Download the CSV files from Kaggle: [MNIST CSV Format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
b. Place the files in the `archive` directory:
   - `archive/mnist_train.csv` (for training data)
   - `archive/mnist_test.csv` (for test data)

## Implementation Details

### Core Components

1. **Tensor Operations** (`mytorch/__init__.py`)
   - Basic arithmetic operations (+, -, *, /)
   - Matrix operations (dot product, transpose)
   - Automatic differentiation with backward pass
   - Gradient computation and accumulation

2. **Neural Network Modules** (`mytorch/nn.py`)
   - Linear layer with weights and biases
   - Activation functions (ReLU)
   - Loss functions (Cross Entropy)
   - Forward and backward propagation

3. **Optimizers**
   - SGD (Stochastic Gradient Descent)
   - Adam optimizer implementation

### Architecture

The framework follows a modular design similar to PyTorch:

```
MyTorch/
├── mytorch/
│   ├── __init__.py    # Core tensor operations
│   └── nn.py          # Neural network modules
├── mnist_example.py    # Training script
├── evaluate_mnist.py   # Evaluation script
└── requirements.txt    # Dependencies
```

## Usage

### Training

Run the training script:
```bash
python mnist_example.py
```

The script will:
1. Load and preprocess MNIST data
2. Create a neural network model
3. Train for specified epochs
4. Save model weights

### Evaluation

Run the evaluation script:
```bash
python evaluate_mnist.py
```

## Performance Results

### Overall Metrics
- Accuracy: 97.16%
- Macro Precision: 97.23%
- Macro Recall: 97.14%
- Macro F1-score: 97.15%

### Per-class Performance

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 98.28%    | 98.98% | 98.63%   |
| 1     | 97.84%    | 99.56% | 98.69%   |
| 2     | 98.87%    | 93.60% | 96.17%   |
| 3     | 90.64%    | 98.81% | 94.55%   |
| 4     | 97.95%    | 97.45% | 97.70%   |
| 5     | 96.26%    | 98.09% | 97.17%   |
| 6     | 98.11%    | 97.49% | 97.80%   |
| 7     | 97.86%    | 97.67% | 97.76%   |
| 8     | 98.28%    | 93.63% | 95.90%   |
| 9     | 98.18%    | 96.13% | 97.15%   |

### Confusion Matrix
```
[[ 970    1    0    2    0    0    4    0    2    1]
 [   0 1130    0    3    0    1    1    0    0    0]
 [   3    7  966   39    3    0    2    6    6    0]
 [   0    0    3  998    0    6    0    0    1    2]
 [   1    1    1    1  957    0    5    6    1    9]
 [   2    0    0    9    2  875    1    1    1    1]
 [   3    3    0    2    4   11  934    0    1    0]
 [   2    5    6    5    0    0    0 1004    2    4]
 [   4    3    1   29    6   12    3    3  912    1]
 [   2    5    0   13    5    4    2    6    2  970]]
```

### Key Observations

1. **Strong Overall Performance**
   - 97.16% accuracy on test set
   - Consistent performance across classes
   - High precision and recall for most digits

2. **Per-class Analysis**
   - Best performance on digit 1 (F1: 98.69%)
   - Most challenging: digit 3 (F1: 94.55%)
   - Very high precision across all classes (>90%)

3. **Error Patterns**
   - Most confusion between visually similar digits (2↔3, 4↔9)
   - Minimal confusion between dissimilar digits
   - Balanced error distribution

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes. Areas for potential enhancement:

1. Additional layer types (Conv2D, MaxPool, etc.)
2. More optimization algorithms
3. Data augmentation techniques
4. Support for other datasets
5. Performance optimizations

## License

MIT License - feel free to use this code for educational purposes.
