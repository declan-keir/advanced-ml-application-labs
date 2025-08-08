import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pop_target(df, target_col):
    """
    Extract target variable from input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        tuple: (features, target) where features is df without target column
               and target is the target column
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    target = df[target_col].copy()
    features = df.drop(columns=[target_col]).copy()
    
    return features, target


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path="./data"):
    """
    Save training, validation and testing sets locally with numpy.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features  
        y_val (pd.Series): Validation target
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        path (str): Path to folder for saving files
        
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    if X_train is not None and y_train is not None:
        np.save(os.path.join(path, 'X_train.npy'), X_train.values)
        np.save(os.path.join(path, 'y_train.npy'), y_train.values)
    
    if X_val is not None and y_val is not None:
        np.save(os.path.join(path, 'X_val.npy'), X_val.values)
        np.save(os.path.join(path, 'y_val.npy'), y_val.values)
    
    if X_test is not None and y_test is not None:
        np.save(os.path.join(path, 'X_test.npy'), X_test.values)
        np.save(os.path.join(path, 'y_test.npy'), y_test.values)


def load_sets(path="./data"):
    """
    Load locally saved training, validation and testing sets.
    
    Args:
        path (str): Path to folder containing saved files
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test) or None for missing sets
    """
    sets = {}
    
    # Define file mappings
    files = {
        'X_train': 'X_train.npy',
        'y_train': 'y_train.npy', 
        'X_val': 'X_val.npy',
        'y_val': 'y_val.npy',
        'X_test': 'X_test.npy',
        'y_test': 'y_test.npy'
    }
    
    # Load files if they exist
    for key, filename in files.items():
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            sets[key] = np.load(filepath)
        else:
            sets[key] = None
    
    return sets['X_train'], sets['y_train'], sets['X_val'], sets['y_val'], sets['X_test'], sets['y_test']


def subsets_x_y(features, target, start_index, end_index):
    """
    Subset features and target based on specified indexes.
    
    Args:
        features (pd.DataFrame): Input features dataframe
        target (pd.Series): Input target series
        start_index (int): Starting index for subset
        end_index (int): Ending index for subset
        
    Returns:
        tuple: (subsetted_features, subsetted_target)
    """
    if start_index < 0 or end_index > len(features):
        raise ValueError(f"Index range [{start_index}:{end_index}] is out of bounds for dataframe of length {len(features)}")
    
    if start_index >= end_index:
        raise ValueError("Start index must be less than end index")
    
    subsetted_features = features.iloc[start_index:end_index].copy()
    subsetted_target = target.iloc[start_index:end_index].copy()
    
    return subsetted_features, subsetted_target


def split_sets_time(df, target_col, test_ratio=0.2):
    """
    Split dataframe into time-based training, validation and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Column name of the target variable
        test_ratio (float): Ratio for splitting (default: 0.2)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Extract features and target
    features, target = pop_target(df, target_col)
    
    # Calculate split points for time-based splitting
    n_samples = len(df)
    test_size = int(n_samples * test_ratio)
    val_size = test_size  # Same size as test set
    train_size = n_samples - test_size - val_size
    
    # Split based on time order (no shuffling)
    X_train = features.iloc[:train_size].copy()
    y_train = target.iloc[:train_size].copy()
    
    X_val = features.iloc[train_size:train_size + val_size].copy()
    y_val = target.iloc[train_size:train_size + val_size].copy()
    
    X_test = features.iloc[train_size + val_size:].copy()
    y_test = target.iloc[train_size + val_size:].copy()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_sets_random(features, target, test_ratio=0.2):
    """
    Split features and target into random training, validation and testing sets.
    
    Args:
        features (pd.DataFrame): Input features dataframe
        target (pd.Series): Target variable
        test_ratio (float): Percentage to be used for testing and validation sets
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Calculate validation ratio to have same number of rows as test set
    # If test_ratio = 0.2, we want 0.2 for test and 0.2 for validation
    # So train_ratio = 0.6, and val_ratio from remaining = 0.25 (0.2/0.8)
    val_ratio_from_train = test_ratio / (1 - test_ratio)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, target, test_size=test_ratio, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_from_train, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test