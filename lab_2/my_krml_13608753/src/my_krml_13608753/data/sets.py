import pandas as pd

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