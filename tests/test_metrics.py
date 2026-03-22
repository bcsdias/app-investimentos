import pandas as pd
import numpy as np
import pytest
from src.engine.metrics import calculate_drawdown, calculate_rolling_volatility, calculate_rolling_sharpe

def test_drawdown_max():
    data = {
        'Ativo1': [100.0, 110.0, 99.0, 120.0],
        'Ativo2': [100.0, 90.0, 80.0, 70.0]
    }
    df = pd.DataFrame(data)
    dd = calculate_drawdown(df)
    
    # Ativo 1:
    # 100 -> DD=0
    # 110 -> DD=0 (Highest = 110)
    # 99 -> DD= (99/110) - 1 = -0.1 (-10%)
    # 120 -> DD=0 (Highest = 120)
    np.testing.assert_almost_equal(dd['Ativo1'].iloc[2], -0.1)
    np.testing.assert_almost_equal(dd['Ativo1'].min(), -0.1)
    
    # Ativo 2: Só cai. Highest is sempre 100.
    # 70/100 - 1 = -0.3
    np.testing.assert_almost_equal(dd['Ativo2'].min(), -0.3)

def test_volatility_annualisation():
    # Cria uma série artificial com 1% de retorno diário fixo
    # Como não há desvio na média (é fixo 1%), o std = 0. Vol = 0
    returns = np.full(300, 0.01)
    prices = 100 * (1 + returns).cumprod()
    df = pd.DataFrame({'Fake': prices})
    
    vol = calculate_rolling_volatility(df, window=252)
    # O std deve ser muito baixo
    assert vol['Fake'].dropna().max() < 1e-10

def test_sharpe_sign():
    # Ativo 1 cresce 1% ao dia e a RF é 10% a.a. (muito menor que 1% a.d.) -> Sharpe Positivo
    # Ativo 2 não cresce (0%) -> Sharpe Negativo
    dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
    prices1 = 100 * (1 + np.full(300, 0.01)).cumprod()
    prices2 = np.full(300, 100.0)
    
    df = pd.DataFrame({'A1': prices1, 'A2': prices2}, index=dates)
    
    sharpe = calculate_rolling_sharpe(df, rf_constant=0.10, window=252)
    
    # Verifica o último valor na janela de 252 (tem dados suficientes)
    assert sharpe['A1'].iloc[-1] > 0
    assert sharpe['A2'].iloc[-1] < 0
