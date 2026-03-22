import numpy as np
import pandas as pd

def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Drawdown histórico (queda percentual em relação à máxima histórica).
    
    Args:
        df: DataFrame com as séries temporais de preços/índices de ativos.
        
    Returns:
        pd.DataFrame com a evolução do Drawdown para todos os ativos no mesmo índice.
    """
    if df.empty:
        return pd.DataFrame()
        
    rolling_max = df.cummax()
    drawdown = (df / rolling_max) - 1.0
    return drawdown

def calculate_rolling_volatility(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calcula a volatilidade contínua (anualizada) baseada em uma janela móvel de dias úteis.
    
    Args:
        df: DataFrame histórico dos ativos (preços ou índices).
        window: Número de observações na janela (padrão 252 dias úteis).
        
    Returns:
        pd.DataFrame de volatidade na mesma medida temporal após pular os NaN.
    """
    if df.empty:
        return pd.DataFrame()

    daily_ret = df.pct_change()
    rolling_vol = daily_ret.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol.dropna(how='all')

def calculate_rolling_sharpe(df: pd.DataFrame, rf_series: pd.Series = None, rf_constant: float = 0.10, window: int = 252) -> pd.DataFrame:
    """
    Calcula o Sharpe Ratio contínuo baseado em janela móvel.

    Args:
        df: DataFrame base com os ativos.
        rf_series: Série temporal (índice base) da taxa livre de risco, ex: histórico da SELIC (opcional).
        rf_constant: Taxa anual fixa caso rf_series não exista (ex: 0.10).
        window: Janela temporal a calcular o valor.
        
    Returns:
        pd.DataFrame de Sharp ratio.
    """
    if df.empty:
        return pd.DataFrame()
        
    daily_ret = df.pct_change()
    rf_daily_series = pd.Series(0.0, index=daily_ret.index)
    
    if rf_series is not None and not rf_series.empty:
         # Trabalha diferença do benchmark
         selic_daily = rf_series.pct_change().fillna(0)
         rf_daily_series = selic_daily.reindex(daily_ret.index).ffill().fillna(0)
    else:
         rf_daily_series[:] = ( (1.0 + rf_constant) ** (1/252) ) - 1.0

    excess_ret = daily_ret.sub(rf_daily_series, axis=0)
    
    rolling_mean = excess_ret.rolling(window=window).mean()
    rolling_std = excess_ret.rolling(window=window).std()
    
    # Média Diária / Vol Diária * sqrt(252)
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    return rolling_sharpe.dropna(how='all')
