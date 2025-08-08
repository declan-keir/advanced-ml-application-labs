import pytest
import pandas as pd
from my_krml_13608753.data.sets import pop_target


def test_pop_target_basic():
    # Create test dataframe
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6], 
        'target': [7, 8, 9]
    })
    
    features, target = pop_target(df, 'target')
    
    assert list(features.columns) == ['feature1', 'feature2']
    assert target.name == 'target'
    assert len(features) == 3
    assert len(target) == 3

def test_pop_target_missing_column():
    df = pd.DataFrame({'feature1': [1, 2, 3]})
    
    with pytest.raises(ValueError):
        pop_target(df, 'nonexistent')