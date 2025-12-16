"""
Script 04: Keras Sequential Model Examples
Various examples of Keras Sequential models for the lab
"""
import tensorflow as tf
import numpy as np

def example_1_basic_sequential():
    """Basic Sequential model example"""
    print("\n" + "="*50)
    print("EXAMPLE 1: BASIC SEQUENTIAL MODEL")
    print("="*50)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    return model

def example_2_different_activations():
    """Model with different activation functions"""
    print("\n" + "="*50)
    print("EXAMPLE 2: DIFFERENT ACTIVATION FUNCTIONS")
    print("="*50)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    return model

def example_3_dropout_layers():
    """Model with dropout for regularization"""
    print("\n" + "="*50)
    print("EXAMPLE 3: MODEL WITH DROPOUT LAYERS")
    print("="*50)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(50,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    return model

def example_4_batch_normalization():
    """Model with batch normalization"""
    print("\n" + "="*50)
    print("EXAMPLE 4: MODEL WITH BATCH NORMALIZATION")
    print("="*50)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(30,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    return model

def example_5_mushroom_model(input_shape):
    """Complete mushroom classification model"""
    print("\n" + "="*50)
    print("EXAMPLE 5: MUSHROOM CLASSIFICATION MODEL")
    print("="*50)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # First hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Third hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

if __name__ == "__main__":
    print("Keras Sequential Model Examples")
    
    # Run all examples
    example_1_basic_sequential()
    example_2_different_activations()
    example_3_dropout_layers()
    example_4_batch_normalization()
    
    # For mushroom model, we need input shape
    print("\n" + "="*50)
    print("TRYING MUSHROOM MODEL (NEEDS DATA)")
    print("="*50)
    
    # Try to load data to get input shape
    try:
        X_train = np.load("data/X_train.npy", allow_pickle=True)
        input_shape = X_train.shape[1]
        example_5_mushroom_model(input_shape)
    except FileNotFoundError:
        print("Preprocessed data not found. Run 02_data_preprocessing.py first.")
