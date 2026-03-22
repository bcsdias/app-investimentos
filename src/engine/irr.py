import numpy as np
import pandas as pd
from typing import Optional

def calculate_xirr(cash_flows: list[float] | np.ndarray, dates: list[str] | pd.DatetimeIndex) -> Optional[float]:
    """
    Calcula a Taxa Interna de Retorno (XIRR/TIR) usando o método de Newton-Raphson.
    
    Args:
        cash_flows: Lista ou array de fluxos de caixa (negativo = investimento, positivo = resgate/valor atual).
        dates: Lista ou DatetimeIndex das datas correspondentes aos fluxos.
        
    Returns:
        O valor da taxa interna de retorno como float (ex: 0.1 para 10%), ou None se não convergir.
    """
    if len(cash_flows) < 2: return None
    
    # Garante tipos numpy/pandas
    cash_flows = np.array(cash_flows)
    dates = pd.to_datetime(dates)
    
    # Datas relativas em anos
    start_date = dates[0]
    days = (dates - start_date).days.values
    years = days / 365.0
    
    # Chute inicial (10%)
    r = 0.1
    
    for _ in range(50): # Max iterações
        if r <= -1.0: r = -0.99
        
        # NPV = sum(Flow / (1+r)^Year)
        factor = (1 + r) ** years
        npv = np.sum(cash_flows / factor)
        
        # Derivada: d/dr [ C * (1+r)^-y ] = C * -y * (1+r)^(-y-1)
        d_npv = np.sum(-years * cash_flows / ((1 + r) ** (years + 1)))
        
        if abs(npv) < 1e-5:
            return r
        
        if d_npv == 0:
            return None
            
        new_r = r - npv / d_npv
        
        if abs(new_r - r) < 1e-5:
            return new_r
            
        r = new_r
        
    return r if abs(npv) < 0.1 else None
