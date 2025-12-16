"""
Script 02: Data Preprocessing
Clean and prepare the mushroom dataset for neural network training
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# Setup
DATA_PATH = Path("data/mushrooms.csv")

def load_and_preprocess():
    """Load data and perform basic preprocessing"""
    df = pd.read_csv(DATA_PATH)
    
    print("Original shape:", df.shape)
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # For categorical data, we can fill with mode
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def encode_categorical_data(df):
    """Encode categorical variables for neural network"""
    print("\n" + "="*50)
    print("ENCODING CATEGORICAL DATA")
    print("="*50)
    
    # Separate features and target
    # Assuming 'class' column is the target (e: edible, p: poisonous)
    target_col = 'class'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nTarget encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # One-hot encode features (for categorical data)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"\nFeatures after one-hot encoding: {X_encoded.shape}")
    print(f"Number of features: {X_encoded.shape[1]}")
    
    return X_encoded, y_encoded, label_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Training target: {y_train.shape}")
    print(f"Testing target: {y_test.shape}")
    
    # Check class distribution in splits
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"\nClass distribution in training set: {dict(zip(unique_train, counts_train))}")
    print(f"Class distribution in testing set: {dict(zip(unique_test, counts_test))}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Starting data preprocessing...")
    
    # Load data
    df = load_and_preprocess()
    
    # Encode categorical data
    X, y, label_encoder = encode_categorical_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save processed data for later use
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    
    print("\nPreprocessing complete!")
    print("Saved processed data to data/directory")
