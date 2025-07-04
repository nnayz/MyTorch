import numpy as np
import pandas as pd
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch import tensor
from tqdm import tqdm


def load_mnist_data(train_path, test_path=None):
    """Load MNIST data from CSV files"""
    print("Loading training data...")
    train_data = pd.read_csv(train_path)
    
    # Split features and labels
    X_train = train_data.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize to [0,1]
    y_train = train_data.iloc[:, 0].values.astype(np.int32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    if test_path:
        print("Loading test data...")
        test_data = pd.read_csv(test_path)
        X_test = test_data.iloc[:, 1:].values.astype(np.float32) / 255.0
        y_test = test_data.iloc[:, 0].values.astype(np.int32)
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train


def create_model(input_size=784, hidden_size=128, num_classes=10):
    """Create a simple feedforward neural network"""
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )
    return model


def get_batch(X, y, batch_size, start_idx):
    """Get a batch of data"""
    end_idx = min(start_idx + batch_size, len(X))
    batch_X = X[start_idx:end_idx]
    batch_y = y[start_idx:end_idx]
    return batch_X, batch_y


def calculate_accuracy(model, X, y, batch_size=100):
    """Calculate accuracy on a dataset"""
    correct = 0
    total = 0
    
    model.eval()  # Set to evaluation mode
    
    num_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
    
    with tqdm(range(0, len(X), batch_size), desc="Evaluating", leave=False, disable=len(X) < 1000) as pbar:
        for i in pbar:
            batch_X, batch_y = get_batch(X, y, batch_size, i)
            
            # Forward pass
            X_tensor = tensor(batch_X, requires_grad=False)
            outputs = model(X_tensor)
            
            # Get predictions
            predictions = np.argmax(outputs.data, axis=1)
            
            # Count correct predictions
            correct += np.sum(predictions == batch_y)
            total += len(batch_y)
            
            # Update progress bar
            if total > 0:
                pbar.set_postfix({'acc': f'{correct/total:.4f}'})
    
    model.train()  # Set back to training mode
    return correct / total


def train_model(model, X_train, y_train, X_test=None, y_test=None, 
                epochs=10, batch_size=32, learning_rate=0.001):
    """Train the model"""
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_batches = len(X_train) // batch_size
    
    print(f"Starting training...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Number of batches per epoch: {num_batches}")
    print("-" * 60)
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0.0
        
        # Shuffle data for each epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Create progress bar for batches
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx in batch_pbar:
            start_idx = batch_idx * batch_size
            batch_X, batch_y = get_batch(X_train_shuffled, y_train_shuffled, batch_size, start_idx)
            
            # Convert to tensors
            X_tensor = tensor(batch_X, requires_grad=False)
            y_tensor = tensor(batch_y, requires_grad=False)
            
            # Forward pass
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            
            # Update progress bar with current loss
            current_avg_loss = total_loss / (batch_idx + 1)
            batch_pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'batch': f'{batch_idx+1}/{num_batches}'
            })
        
        # Calculate epoch statistics
        avg_loss = total_loss / num_batches
        
        # Calculate accuracy on training set (subset for speed)
        train_acc = calculate_accuracy(model, X_train[:1000], y_train[:1000], batch_size=100)
        
        # Calculate test accuracy if test data is provided
        test_acc = None
        if X_test is not None and y_test is not None:
            test_acc = calculate_accuracy(model, X_test, y_test, batch_size=100)
        
        # Print epoch summary
        epoch_summary = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}"
        if test_acc is not None:
            epoch_summary += f", Test Acc: {test_acc:.4f}"
        print(epoch_summary)
    
    print("Training completed!")

    # Save model weights
    weight_dict = {}
    for name, param in model.named_parameters():
        weight_dict[name] = param.data
    np.savez('mnist_weights.npz', **weight_dict)
    print("Model weights saved to 'mnist_weights.npz'.")


def main():
    """Main training function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    try:
        # Try to load both train and test data
        X_train, y_train, X_test, y_test = load_mnist_data(
            'archive/mnist_train.csv', 
            'archive/mnist_test.csv'
        )
    except FileNotFoundError:
        try:
            # If test file doesn't exist, just load training data
            X_train, y_train = load_mnist_data('archive/mnist_train.csv')
            X_test, y_test = None, None
            print("Test data not found, using training data only")
        except FileNotFoundError:
            print("Error: Could not find MNIST data files!")
            print("Please ensure 'archive/mnist_train.csv' exists in the current directory")
            return
    
    # Create model
    model = create_model()
    print(f"Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train model
    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=5,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Final evaluation
    if X_test is not None and y_test is not None:
        final_test_acc = calculate_accuracy(model, X_test, y_test)
        print(f"Final Test Accuracy: {final_test_acc:.4f}")
    
    # Show some predictions
    print("\nSample predictions:")
    model.eval()
    sample_size = 10
    sample_X = X_test[:sample_size] if X_test is not None else X_train[:sample_size]
    sample_y = y_test[:sample_size] if y_test is not None else y_train[:sample_size]
    
    X_sample_tensor = tensor(sample_X, requires_grad=False)
    outputs = model(X_sample_tensor)
    predictions = np.argmax(outputs.data, axis=1)
    
    for i in range(sample_size):
        print(f"True: {sample_y[i]}, Predicted: {predictions[i]}")


if __name__ == "__main__":
    main() 