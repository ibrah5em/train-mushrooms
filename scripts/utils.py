"""
Utility functions for mushroom classification lab
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}")

def print_tensorflow_info():
    """Print TensorFlow version and device info"""
    print("="*50)
    print("TENSORFLOW INFORMATION")
    print("="*50)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available: Yes ({len(gpus)} device(s))")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("GPU Available: No (using CPU)")
    
    # Check eager execution
    print(f"Eager Execution: {tf.executing_eagerly()}")

def save_model_architecture(model, filepath):
    """Save model architecture as JSON"""
    model_json = model.to_json()
    
    with open(filepath, 'w') as json_file:
        json_file.write(model_json)
    
    print(f"Model architecture saved to {filepath}")

def load_model_architecture(filepath):
    """Load model architecture from JSON"""
    with open(filepath, 'r') as json_file:
        model_json = json_file.read()
    
    model = tf.keras.models.model_from_json(model_json)
    print(f"Model architecture loaded from {filepath}")
    return model

def save_training_history(history, filepath):
    """Save training history to file"""
    # Convert history to dictionary
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(value) for value in values]
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {filepath}")

def load_training_history(filepath):
    """Load training history from file"""
    with open(filepath, 'r') as f:
        history_dict = json.load(f)
    
    # Create a simple object to mimic History object
    class HistoryObject:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return HistoryObject(history_dict)

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for the first layer"""
    if len(model.layers) > 0:
        # Get weights from first layer
        weights = model.layers[0].get_weights()[0]
        
        # Calculate feature importance (absolute mean weight)
        importance = np.abs(weights).mean(axis=1)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort and select top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Model has no layers")
        return None

def create_model_summary_file(model, filepath):
    """Save model summary to text file"""
    with open(filepath, 'w') as f:
        # Redirect model summary to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model summary saved to {filepath}")

def check_data_leakage(X_train, X_test, threshold=0.95):
    """Check for data leakage between train and test sets"""
    print("\n" + "="*50)
    print("CHECKING FOR DATA LEAKAGE")
    print("="*50)
    
    # Convert to DataFrames if they aren't already
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
    else:
        X_train_df = X_train
        X_test_df = X_test
    
    # Check for duplicate samples
    train_set = set([tuple(row) for row in X_train_df.values])
    test_set = set([tuple(row) for row in X_test_df.values])
    
    duplicates = train_set.intersection(test_set)
    duplicate_percentage = len(duplicates) / len(X_test_df) * 100
    
    print(f"Number of duplicate samples between train and test: {len(duplicates)}")
    print(f"Percentage of test samples that appear in train: {duplicate_percentage:.2f}%")
    
    if duplicate_percentage > threshold:
        print("WARNING: Potential data leakage detected!")
        print(f"   More than {threshold}% of test samples appear in training data.")
    else:
        print("✓ No significant data leakage detected.")
    
    return len(duplicates), duplicate_percentage

# Initialize on import
print("Utils module loaded")
set_random_seeds()
