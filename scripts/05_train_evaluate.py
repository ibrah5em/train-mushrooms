"""
Script 05: Training and Evaluation
Complete training pipeline with callbacks and evaluation
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=['Edible', 'Toxic']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def evaluate_comprehensive(model, X_test, y_test, label_encoder=None):
    """Comprehensive model evaluation"""
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Classification report
    print("\nClassification Report:")
    print("="*50)
    
    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = ['Class 0', 'Class 1']
    
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred, target_names)
    
    # Additional metrics
    from sklearn.metrics import roc_auc_score, roc_curve
    
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'confusion_matrix': cm,
        'auc_score': auc_score
    }

if __name__ == "__main__":
    print("Loading data...")
    
    # Load data
    X_train = np.load("data/X_train.npy", allow_pickle=True)
    X_test = np.load("data/X_test.npy", allow_pickle=True)
    y_train = np.load("data/y_train.npy", allow_pickle=True)
    y_test = np.load("data/y_test.npy", allow_pickle=True)
    
    # Create a simple model for demonstration
    input_shape = X_train.shape[1]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("\nTraining model with callbacks...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate comprehensively
    evaluate_comprehensive(model, X_test, y_test)
    
    print("\nTraining and evaluation complete!")
    print("Check the generated plots in the project root directory.")
