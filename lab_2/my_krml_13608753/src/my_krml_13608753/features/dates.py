import numpy as np
import pandas as pd


def convert_to_date(df, cols):
    """
    Convert specified columns in dataframe to datetime.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cols (list): List of column names to convert
        
    Returns:
        pd.DataFrame: Transformed dataframe with datetime columns
    """
    df_transformed = df.copy()
    
    for col in cols:
        if col not in df_transformed.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        df_transformed[col] = pd.to_datetime(df_transformed[col])
    
    return df_transformed