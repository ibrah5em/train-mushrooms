
"""
Script 01: Data Exploration
Explore the mushroom dataset structure and basic statistics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
DATA_PATH = Path("data/mushrooms.csv")

def load_data():
    """Load and display basic info about the dataset"""
    df = pd.read_csv(DATA_PATH)
    
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n" + "="*50)
    print("COLUMN INFORMATION")
    print("="*50)
    print(df.info())
    
    print("\n" + "="*50)
    print("BASIC STATISTICS")
    print("="*50)
    print(df.describe(include='all'))
    
    return df

def check_target_distribution(df):
    """Check distribution of target variable (edible vs toxic)"""
    target_col = 'class'  # Assuming 'class' is the target
    if target_col in df.columns:
        print("\n" + "="*50)
        print("TARGET DISTRIBUTION")
        print("="*50)
        
        distribution = df[target_col].value_counts()
        print(distribution)
        
        plt.figure(figsize=(8, 5))
        df[target_col].value_counts().plot(kind='bar')
        plt.title('Distribution of Mushroom Classes')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('figures/target_distribution.png')
        plt.show()
        
        return distribution
    else:
        print(f"Warning: Column '{target_col}' not found in dataset")
        print("Available columns:", df.columns.tolist())
        return None

def check_missing_values(df):
    """Check for missing values in the dataset"""
    print("\n" + "="*50)
    print("MISSING VALUES CHECK")
    print("="*50)
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage (%)': missing_percent
    }).sort_values('Missing Values', ascending=False)
    
    print(missing_df[missing_df['Missing Values'] > 0])
    
    if missing.sum() == 0:
        print("No missing values found!")
    
    return missing_df

def explore_categorical_features(df, n=10):
    """Explore categorical features"""
    print("\n" + "="*50)
    print("CATEGORICAL FEATURES EXPLORATION")
    print("="*50)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"Found {len(categorical_cols)} categorical columns:")
    for col in categorical_cols:
        unique_values = df[col].nunique()
        print(f"  - {col}: {unique_values} unique values")
        
        if unique_values <= n:
            print(f"    Values: {df[col].unique().tolist()}")

if __name__ == "__main__":
    print("Starting data exploration...")
    df = load_data()
    check_target_distribution(df)
    check_missing_values(df)
    explore_categorical_features(df)
    print("\nExploration complete!")
