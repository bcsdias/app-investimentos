import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import brentq


def calculate_xirr(
    cash_flows: list[float] | np.ndarray,
    dates: list[str] | pd.DatetimeIndex,
) -> Optional[float]:
    """
    Calcula a Taxa Interna de Retorno (XIRR/TIR) anualizada.
    
    Ordena os fluxos por data e conta o tempo em anos usando convenção Actual/365
    em relação à data mais antiga.
    
    Critério de convergência invariante à escala: |NPV| <= 1e-9 * sum(|cash_flows|).
    
    Tentativa inicial via Newton-Raphson a partir de r = 0.1. Em caso de falha ou
    divergência, utiliza bracket solver (Brent) no intervalo [-0.999999, 1e3],
    expandindo o teto se necessário.
    
    Args:
        cash_flows: Lista ou array de fluxos de caixa (negativo = investimento, positivo = resgate/valor atual).
        dates: Lista ou DatetimeIndex das datas correspondentes aos fluxos.
        
    Returns:
        O valor da taxa interna de retorno como float (ex: 0.1 para 10%), ou None se não convergir.
    """
    if cash_flows is None or dates is None:
        return None

    try:
        cash_flows_arr = np.asarray(cash_flows, dtype=float)
        dates_dt = pd.to_datetime(dates)
    except Exception:
        return None

    if len(cash_flows_arr) < 2 or len(dates_dt) < 2 or len(cash_flows_arr) != len(dates_dt):
        return None

    if np.isnan(cash_flows_arr).any() or pd.isna(dates_dt).any():
        return None

    # Se todos os fluxos tiverem o mesmo sinal -> None
    has_pos = bool(np.any(cash_flows_arr > 0))
    has_neg = bool(np.any(cash_flows_arr < 0))
    if not (has_pos and has_neg):
        return None

    # Ordena por data
    dates_idx = pd.DatetimeIndex(dates_dt)
    date_vals = dates_idx.values
    order = np.argsort(date_vals)
    sorted_dates = date_vals[order]
    sorted_flows = cash_flows_arr[order]

    # Se houver apenas uma data distinta -> None
    if sorted_dates[0] == sorted_dates[-1]:
        return None

    # Anos = dias desde a data mais antiga / 365 (Actual/365 contado do fluxo mais antigo)
    days = (sorted_dates - sorted_dates[0]) / np.timedelta64(1, "D")
    years = days / 365.0

    # Invariância à escala: |npv| <= 1e-9 * sum(|cash_flows|)
    total_abs_flow = float(np.sum(np.abs(sorted_flows)))
    if total_abs_flow == 0.0:
        return None
    tol = 1e-9 * total_abs_flow

    def npv(rate: float) -> float:
        if rate <= -1.0:
            return float("nan")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            factors = (1.0 + rate) ** years
            terms = np.where(np.isinf(factors), 0.0, sorted_flows / factors)
            val = np.sum(terms)
            return float(val) if np.isfinite(val) else float("nan")

    def d_npv(rate: float) -> float:
        if rate <= -1.0:
            return float("nan")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            factors = (1.0 + rate) ** (years + 1.0)
            terms = np.where(np.isinf(factors), 0.0, -years * sorted_flows / factors)
            val = np.sum(terms)
            return float(val) if np.isfinite(val) else float("nan")

    # 1. Tentativa inicial com Newton-Raphson a partir de taxa r = 0.1
    r = 0.1
    for _ in range(50):
        val = npv(r)
        if not np.isfinite(val):
            break
        d = d_npv(r)
        if abs(val) <= tol:
            if not np.isfinite(d) or d == 0.0:
                return float(r)
            step = val / d
            if abs(step) <= 1e-9 * max(abs(r), 1.0):
                return float(r)
            refined_r = r - step
            if np.isfinite(refined_r) and refined_r > -0.999999:
                refined_val = npv(refined_r)
                if np.isfinite(refined_val) and abs(refined_val) <= tol:
                    return float(refined_r)
            return float(r)

        if not np.isfinite(d) or d == 0.0:
            break
        step = val / d
        new_r = r - step
        if not np.isfinite(new_r) or new_r <= -0.999999:
            break
        r = new_r
    else:
        val = npv(r)
        if np.isfinite(val) and abs(val) <= tol:
            return float(r)

    # 2. Fallback por bracket solver (brentq) num intervalo com troca de sinal
    lower = -0.999999
    upper = 1000.0

    f_lower = npv(lower)
    f_upper = npv(upper)

    if not np.isfinite(f_lower) or not np.isfinite(f_upper):
        return None

    if abs(f_lower) <= tol:
        return float(lower)
    if abs(f_upper) <= tol:
        return float(upper)

    # Expande o teto se necessário para encontrar troca de sinal
    if f_lower * f_upper > 0:
        for _ in range(120):
            upper *= 10.0
            f_upper = npv(upper)
            if not np.isfinite(f_upper):
                break
            if abs(f_upper) <= tol:
                return float(upper)
            if f_lower * f_upper <= 0:
                break
        else:
            return None

    if f_lower * f_upper > 0 or not (np.isfinite(f_lower) and np.isfinite(f_upper)):
        return None

    try:
        root = brentq(npv, lower, upper, maxiter=200)
        if abs(npv(root)) <= tol:
            return float(root)
    except Exception:
        pass

    return None
