import numpy as np
import pandas as pd

FLOW_COLUMNS: tuple[str, ...] = ("vlr_mercado", "vlr_investido", "proventos", "lucro")


def group_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa a carteira por data somando as colunas de fluxo e mercado.

    Colunas produzidas:
        'date' (datetime64) + FLOW_COLUMNS (+ 'fluxo' se presente na entrada),
        ordenadas cronologicamente e com colunas ausentes preenchidas com 0.0.
    """
    cols = ["date"] + list(FLOW_COLUMNS)
    has_fluxo = df is not None and "fluxo" in df.columns
    if has_fluxo:
        cols.append("fluxo")

    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    df = df.copy()

    # Normalização da coluna de data
    if "date" not in df.columns:
        if df.index.name and df.index.name.lower() in ["date", "data"]:
            df = df.reset_index()
            if "date" not in df.columns:
                for col in df.columns:
                    if str(col).lower() in ["date", "data"]:
                        df = df.rename(columns={col: "date"})
                        break
        else:
            for col in df.columns:
                if str(col).lower() in ["date", "data"]:
                    df = df.rename(columns={col: "date"})
                    break

    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(
                columns={df.index.name: "date"} if df.index.name else {df.columns[0]: "date"}
            )
        else:
            return pd.DataFrame(columns=cols)

    df["date"] = pd.to_datetime(df["date"])

    agg_cols = list(FLOW_COLUMNS)
    if has_fluxo:
        agg_cols.append("fluxo")

    for col in agg_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    grp = (
        df.groupby("date", as_index=False)[agg_cols]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return grp


def external_flows(grp: pd.DataFrame) -> pd.Series:
    """
    Calcula o fluxo de caixa externo líquido por período.

    Fórmula:
        fluxo = Δvlr_investido − lucro − proventos
        Na linha 0: Δvlr_investido = vlr_investido
    """
    if grp is None or grp.empty:
        return pd.Series(dtype=float, name="fluxo")

    vlr_investido = (
        grp["vlr_investido"]
        if "vlr_investido" in grp.columns
        else pd.Series(0.0, index=grp.index)
    )
    delta_investido = vlr_investido.diff()
    if len(delta_investido) > 0:
        delta_investido.iloc[0] = vlr_investido.iloc[0]

    lucro = grp["lucro"] if "lucro" in grp.columns else 0.0
    proventos = grp["proventos"] if "proventos" in grp.columns else 0.0

    fluxo = delta_investido - lucro - proventos
    fluxo.name = "fluxo"
    return fluxo


def hpr(vlr_mercado: pd.Series, fluxo: pd.Series) -> pd.Series:
    """
    Calcula o Holding Period Return (HPR) mensal/periódico da DLP.

    Fórmula:
        r = (M_t − min(F, 0)) / (M_{t−1} + max(F, 0))
        Linha 0 e denominador 0 → 1.0
    """
    if vlr_mercado is None or vlr_mercado.empty:
        return pd.Series(dtype=float, name="hpr")

    m_t = vlr_mercado.astype(float)
    f = fluxo.astype(float)
    m_prev = m_t.shift(1).fillna(0.0)

    num = m_t - np.minimum(f, 0.0)
    den = m_prev + np.maximum(f, 0.0)

    res = pd.Series(1.0, index=vlr_mercado.index, dtype=float, name="hpr")
    nonzero = den != 0.0
    res[nonzero] = num[nonzero] / den[nonzero]
    res = res.fillna(1.0)

    if len(res) > 0:
        res.iloc[0] = 1.0

    return res
