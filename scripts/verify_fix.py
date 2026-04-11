import pandas as pd
import numpy as np
import sys
import os

# Set base dir
BASE_DIR = r"c:\onedrive-bcsdias\OneDrive\dev\app_investimentos"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.engine.twr import calculate_twr

def test_twr_robustness():
    print("Testing TWR robustness with different date column names...")
    
    # Case 1: 'date' (Standard)
    df1 = pd.DataFrame({
        'date': ['2023-01-01', '2023-02-01'],
        'vlr_mercado': [100, 110],
        'vlr_investido': [100, 100],
        'proventos': [0, 0]
    })
    res1 = calculate_twr(df1)
    print(f"Standard 'date': Success. Values: {res1.values}")
    
    # Case 2: 'Data' (Portuguese)
    df2 = pd.DataFrame({
        'Data': ['2023-01-01', '2023-02-01'],
        'vlr_mercado': [100, 110],
        'vlr_investido': [100, 100],
        'proventos': [0, 0]
    })
    res2 = calculate_twr(df2)
    print(f"Portuguese 'Data': Success. Values: {res2.values}")
    
    # Case 3: 'Date' (Capitalized)
    df3 = pd.DataFrame({
        'Date': ['2023-01-01', '2023-02-01'],
        'vlr_mercado': [100, 110],
        'vlr_investido': [100, 100],
        'proventos': [0, 0]
    })
    res3 = calculate_twr(df3)
    print(f"Capitalized 'Date': Success. Values: {res3.values}")
    
    # Case 4: Index is date
    df4 = pd.DataFrame({
        'vlr_mercado': [100, 110],
        'vlr_investido': [100, 100],
        'proventos': [0, 0]
    }, index=pd.to_datetime(['2023-01-01', '2023-02-01']))
    df4.index.name = 'Data'
    res4 = calculate_twr(df4)
    print(f"Index as 'Data': Success. Values: {res4.values}")

    assert np.allclose(res1.values, res2.values)
    assert np.allclose(res1.values, res3.values)
    assert np.allclose(res1.values, res4.values)
    print("All tests passed!")

if __name__ == "__main__":
    test_twr_robustness()
