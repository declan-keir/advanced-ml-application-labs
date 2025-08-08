import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def print_regressor_scores(y_preds, y_actuals, set_name):
    """
    Print RMSE and MAE scores for regression predictions.
    
    Args:
        y_preds (array-like): Predicted target values
        y_actuals (array-like): Actual target values
        set_name (str): Name of the dataset (e.g., 'Training', 'Validation', 'Test')
        
    Returns:
        None
    """
    if len(y_preds) != len(y_actuals):
        raise ValueError("Predictions and actuals must have the same length")
    
    rmse = np.sqrt(mean_squared_error(y_actuals, y_preds))
    mae = mean_absolute_error(y_actuals, y_preds)
    
    print(f"{set_name} Set Scores:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print()