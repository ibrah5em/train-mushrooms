"""
Script 03: Basic Neural Network
Create and train a simple neural network using TensorFlow
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

def create_basic_model(input_shape):
    """Create a basic neural network model"""
    print("\n" + "="*50)
    print("CREATING BASIC NEURAL NETWORK")
    print("="*50)
    
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Hidden layers
        tf.keras.layers.Dense(128, activation='relu', name='hidden1'),
        tf.keras.layers.Dense(64, activation='relu', name='hidden2'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden3'),
        
        # Output layer (binary classification)
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    print("Model architecture:")
    model.summary()
    
    return model

def compile_model(model):
    """Compile the model with optimizer and loss function"""
    print("\n" + "="*50)
    print("COMPILING MODEL")
    print("="*50)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("Model compiled with:")
    print("  Optimizer: Adam")
    print("  Loss: binary_crossentropy")
    print("  Metrics: accuracy, precision, recall")
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """Train the model"""
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Test F1-Score: {f1_score:.4f}")
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

if __name__ == "__main__":
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    X_train = np.load("data/X_train.npy", allow_pickle=True)
    X_test = np.load("data/X_test.npy", allow_pickle=True)
    y_train = np.load("data/y_train.npy", allow_pickle=True)
    y_test = np.load("data/y_test.npy", allow_pickle=True)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create model
    input_shape = X_train.shape[1]
    model = create_basic_model(input_shape)
    
    # Compile model
    model = compile_model(model)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Save model
    model.save("models/basic_mushroom_model.h5")
    print("\nModel saved to: models/basic_mushroom_model.h5")
