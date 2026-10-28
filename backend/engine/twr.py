import pandas as pd
from engine.flows import external_flows, group_portfolio, hpr


def calculate_twr(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.Series:
    """
    Calcula o Time-Weighted Return (TWR) de uma carteira.

    Args:
        df: DataFrame contendo as colunas 'date', 'vlr_mercado', 'vlr_investido', 'proventos'
            (e opcionalmente 'lucro', 'fluxo').
        start_date: Data de início opcional (formato YYYY-MM-DD).
        end_date: Data de fim opcional (formato YYYY-MM-DD).

    Returns:
        pd.Series contendo o índice TWR (Base 1.0) indexado por data.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Garante cópia para não alterar o original
    df = df.copy()

    # Garante que temos uma coluna 'date'
    if "date" not in df.columns:
        if df.index.name and df.index.name.lower() in ["date", "data"]:
            df = df.reset_index()
            # Se após reset_index o nome não for 'date' (ex: era 'Data'), renomeia
            if "date" not in df.columns:
                for col in df.columns:
                    if col.lower() in ["date", "data"]:
                        df.rename(columns={col: "date"}, inplace=True)
                        break
        else:
            for col in df.columns:
                if col.lower() in ["date", "data"]:
                    df.rename(columns={col: "date"}, inplace=True)
                    break

    if "date" not in df.columns:
        # Se ainda não encontramos, mas o índice é Datetime, usamos ele
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(
                columns={df.index.name: "date"} if df.index.name else {df.columns[0]: "date"}
            )
        else:
            return pd.Series(dtype=float)

    has_fluxo = "fluxo" in df.columns
    df_grp = group_portfolio(df)
    if df_grp.empty:
        return pd.Series(dtype=float)

    if has_fluxo and "fluxo" in df_grp.columns:
        fluxo = df_grp["fluxo"]
    else:
        fluxo = external_flows(df_grp)

    df_grp["hpr"] = hpr(df_grp["vlr_mercado"], fluxo)
    df_grp["twr_index"] = df_grp["hpr"].cumprod()

    # --- Filtro de Período ---
    if start_date:
        df_grp = df_grp[df_grp["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_grp = df_grp[df_grp["date"] <= pd.to_datetime(end_date)]

    if df_grp.empty:
        return pd.Series(dtype=float)

    return df_grp.set_index("date")["twr_index"]
