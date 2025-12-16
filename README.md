# 🍄 Neural Network Lab: Mushroom Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Course**: Neural Networks  
**Professor**: Dr. Tarek Barhoom  
**Project**: Binary Classification of Edible vs Poisonous Mushrooms

---

## 📋 Project Overview

This repository contains a complete machine learning pipeline for classifying mushrooms as edible or poisonous using neural networks. The project demonstrates fundamental concepts in deep learning including data preprocessing, model architecture design, training optimization, and evaluation.

The pipeline is organized into modular scripts that showcase each step of the ML workflow, making it easy to understand and modify for learning purposes.

## 🎯 Learning Objectives

Through this project, you will learn to:
- Perform exploratory data analysis on real-world datasets
- Preprocess categorical data for neural network training
- Build and configure neural network architectures using Keras
- Implement training callbacks for optimization and monitoring
- Evaluate model performance using multiple metrics
- Visualize training progress and results

## 📁 Repository Structure

```
train-mushrooms/
├── data/                          # Dataset and preprocessed arrays
│   ├── mushrooms.csv             # Original mushroom dataset
│   ├── X_train.npy               # Training features
│   ├── X_test.npy                # Test features
│   ├── y_train.npy               # Training labels
│   └── y_test.npy                # Test labels
├── scripts/                       # Python scripts for each pipeline step
│   ├── 01_data_exploration.py    # EDA and data visualization
│   ├── 02_data_preprocessing.py  # Data cleaning and encoding
│   ├── 03_basic_nn.py            # Simple neural network implementation
│   ├── 04_keras_sequential.py    # Keras model examples
│   ├── 05_train_evaluate.py      # Complete training pipeline
│   └── utils.py                  # Helper functions
├── models/                        # Saved trained models
│   ├── basic_mushroom_model.h5
│   └── best_model.h5
├── logs/                          # TensorBoard training logs
├── figures/                       # Generated plots and visualizations
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) virtualenv or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ibrah5em/train-mushrooms
   cd train-mushrooms
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## 📊 Usage Guide

### Step 1: Explore the Data

Run the data exploration script to understand the dataset:

```bash
python scripts/01_data_exploration.py
```

This will display:
- Dataset shape and basic statistics
- Target variable distribution
- Missing values analysis
- Categorical features overview

### Step 2: Preprocess the Data

Clean and prepare the data for training:

```bash
python scripts/02_data_preprocessing.py
```

This script:
- Handles missing values
- Encodes categorical features using one-hot encoding
- Splits data into training and test sets (80/20 split)
- Saves preprocessed NumPy arrays to `data/`

### Step 3: Train a Basic Model

Train a simple neural network:

```bash
python scripts/03_basic_nn.py
```

This demonstrates:
- Basic neural network architecture
- Model compilation with Adam optimizer
- Training loop with validation
- Model evaluation and saving

### Step 4: Explore Different Architectures

Review various Keras Sequential model examples:

```bash
python scripts/04_keras_sequential.py
```

Examples include:
- Basic sequential models
- Models with different activation functions
- Dropout for regularization
- Batch normalization
- Complete mushroom classification architecture

### Step 5: Complete Training Pipeline

Run the full training pipeline with callbacks and comprehensive evaluation:

```bash
python scripts/05_train_evaluate.py
```

Features:
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing (saves best model)
- TensorBoard logging
- Comprehensive evaluation metrics
- Visualization of results

### Monitoring Training with TensorBoard

To visualize training progress in real-time:

```bash
tensorboard --logdir logs --port 6006
```

Then open your browser to `http://localhost:6006`

## 🔍 Using Saved Models

Load and use a trained model for predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best trained model
model = load_model('models/best_model.h5')

# Load test data
X_test = np.load('data/X_test.npy')

# Make predictions
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)

print(f"Predicted classes: {predicted_classes}")
```

## 📈 Expected Results

With the default configuration, you should achieve:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98-99%
- **Precision**: ~98%
- **Recall**: ~98%
- **F1-Score**: ~98%
- **ROC AUC**: ~0.99

*Note: Results may vary slightly due to random initialization*

## 🎓 Assignment Tips

1. **Experiment with architectures**: Try different numbers of layers, neurons, and activation functions
2. **Tune hyperparameters**: Adjust learning rate, batch size, and epochs
3. **Add regularization**: Test different dropout rates and regularization techniques
4. **Compare optimizers**: Try SGD, RMSprop, Adam, and their variants
5. **Document your findings**: Keep notes on what works and what doesn't

## 📚 Key Concepts Covered

- **Data Preprocessing**: One-hot encoding, train-test split, stratification
- **Neural Network Architecture**: Dense layers, activation functions, output layers
- **Training**: Backpropagation, gradient descent, optimization algorithms
- **Regularization**: Dropout, batch normalization, early stopping
- **Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Professor Tarek Barhoom** for course instruction and guidance
- The UCI Machine Learning Repository for the mushroom dataset
- TensorFlow and Keras teams for excellent documentation
- All classmates for collaboration and feedback

---

**Note**: This project is for educational purposes as part of the Neural Networks course. The mushroom dataset is used solely for learning classification techniques and should not be used for actual mushroom identification.

**⚠️ Safety Warning**: Never consume wild mushrooms based on machine learning predictions. Always consult with expert mycologists for mushroom identification.

---

*Last updated: December 2025*
