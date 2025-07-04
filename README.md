# MyTorch

A PyTorch-like deep learning framework implementation for educational purposes. This project includes a complete neural network library with automatic differentiation, common layer types, optimizers, and example implementations.

## Features

- Tensor operations with automatic differentiation
- Neural network modules (Linear, ReLU, etc.)
- Optimizers (SGD, Adam)
- MNIST classification example
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

## Usage

1. Train the model:
```bash
python mnist_example.py
```

2. Evaluate the model:
```bash
python evaluate_mnist.py
```

## Project Structure

- `mytorch/`: Core framework implementation
  - `__init__.py`: Tensor operations
  - `nn.py`: Neural network modules
- `mnist_example.py`: Training script
- `evaluate_mnist.py`: Evaluation script

## Results

The model achieves ~97% accuracy on the MNIST test set after 5 epochs of training.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

MIT License
