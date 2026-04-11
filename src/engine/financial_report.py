import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Importações Refatoradas (Phase 1)
from src.engine.twr import calculate_twr
from src.engine.irr import calculate_xirr
from src.engine.metrics import calculate_drawdown, calculate_rolling_volatility, calculate_rolling_sharpe

# Importação Refatorada (Phase 1)
from src.data.sources.market_data import processar_benchmarks, buscar_historico
from src.data.benchmarks_config import CATALOGO_YF, CATALOGO_B3, CATALOGO_BCB, CATALOGO_TD, BENCHMARKS_ATIVOS
from src.utils.logger import logger
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FinancialReport:
    def __init__(self, output_dir="reports"):
        self.base_output_dir = os.path.join(BASE_DIR, output_dir)
        self.df_combined = pd.DataFrame() 
        self.risk_free_rate = 0.0 
        self.selic_series = None
        self.portfolio_df = None

    def _get_path(self, subfolder, filename):
        folder = os.path.join(self.base_output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def fetch_user_portfolio(self, token, ativo=None, classe=None, start_date=None, end_date=None):
        logger.info("Buscando histórico da carteira do usuário...")
        df = buscar_historico(token, ativo=ativo, classe=classe)
        
        if df is None or df.empty:
            logger.warning("Nenhum dado retornado pela API ou DataFrame vazio.")
            return None
            
        # Garante que temos uma coluna 'date'
        if 'date' not in df.columns:
            if df.index.name and df.index.name.lower() in ['date', 'data']:
                df = df.reset_index()
                if 'date' not in df.columns and 'Data' in df.columns: # Caso reset_index use o nome original
                     df.rename(columns={'Data': 'date'}, inplace=True)
            else:
                for col in df.columns:
                    if col.lower() in ['date', 'data']:
                        df.rename(columns={col: 'date'}, inplace=True)
                        break
        
        if 'date' not in df.columns:
             logger.error("Coluna 'date' não encontrada no DataFrame de histórico.")
             return None

        df['date'] = pd.to_datetime(df['date'])
        df_grp = df.groupby('date')[['vlr_mercado', 'vlr_investido', 'proventos']].sum().reset_index()
        df_grp['fluxo'] = df_grp['vlr_investido'].diff().fillna(df_grp['vlr_investido'].iloc[0]) - df_grp['proventos']
        
        # Ajuste para simulações
        if start_date:
            df_filtered = df_grp[df_grp['date'] >= pd.to_datetime(start_date)].copy()
            if not df_filtered.empty:
                gap = df_filtered['vlr_mercado'].iloc[0] - df_filtered['vlr_investido'].iloc[0]
                df_filtered['vlr_investido'] = df_filtered['vlr_investido'] + gap
            df_grp = df_filtered

        result_series = calculate_twr(df_grp, start_date, end_date)
        
        if not result_series.empty:
            self.portfolio_df = df_grp.copy()
            return result_series
        return None

    def build_dataset(self, user_series=None, years_history=None, active_benchmarks=None, start_date=None, end_date=None):
        if user_series is not None:
            start_date = user_series.index.min().strftime('%Y-%m-%d')
            end_date = user_series.index.max().strftime('%Y-%m-%d')
        elif not start_date or not end_date:
            years = years_history if years_history else 1
            end_dt = pd.Timestamp.today()
            start_dt = end_dt - pd.DateOffset(years=years)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        needed_assets = set()
        carteiras_sinteticas = {}
        
        if 'SELIC' in CATALOGO_BCB:
            needed_assets.add('SELIC')

        benchmarks_to_use = active_benchmarks if active_benchmarks is not None else BENCHMARKS_ATIVOS

        for item in benchmarks_to_use:
            if isinstance(item, str):
                needed_assets.add(item)
            elif isinstance(item, dict):
                nome = item.get('nome')
                comps = item.get('composicao')
                if nome and comps:
                    carteiras_sinteticas[nome] = comps
                    for comp_name in comps.keys():
                        needed_assets.add(comp_name)

        final_needed = set(needed_assets)
        ativos_brl_needed = set()
        for asset in needed_assets:
            if isinstance(asset, str):
                if asset.endswith(' BRL'):
                    base = asset.replace(' BRL', '')
                    final_needed.add(base)
                    ativos_brl_needed.add(base)
                if 'IPCA +' in asset: final_needed.add('IPCA')
                if 'CDI +' in asset or '% do CDI' in asset: final_needed.add('CDI')

        yf_filtered = {k: v for k, v in CATALOGO_YF.items() if k in final_needed}
        b3_filtered = {k: v for k, v in CATALOGO_B3.items() if k in final_needed}
        bcb_filtered = {k: v for k, v in CATALOGO_BCB.items() if k in final_needed}
        td_filtered = {k: v for k, v in CATALOGO_TD.items() if k in final_needed}

        logger.info("Processando Benchmarks...")
        bench_data = processar_benchmarks(
            start_date, end_date,
            yf_filtered, b3_filtered, bcb_filtered,
            td_filtered, carteiras_sinteticas,
            ativos_brl=ativos_brl_needed
        )

        for item in benchmarks_to_use:
            asset_name = item if isinstance(item, str) else item.get('nome')
            if asset_name not in bench_data:
                match = re.match(r"IPCA \+ (\d+(?:\.\d+)?)%", asset_name)
                if match and 'IPCA' in bench_data and bench_data['IPCA'] is not None:
                    s, b = float(match.group(1)), bench_data['IPCA']
                    bench_data[asset_name] = (1 + ((1 + b.pct_change().fillna(0)) * (1 + s/100)**(1/252) - 1)).cumprod() * 100
                
                match = re.match(r"CDI \+ (\d+(?:\.\d+)?)%", asset_name)
                if match and 'CDI' in bench_data and bench_data['CDI'] is not None:
                    s, b = float(match.group(1)), bench_data['CDI']
                    bench_data[asset_name] = (1 + ((1 + b.pct_change().fillna(0)) * (1 + s/100)**(1/252) - 1)).cumprod() * 100

                match = re.match(r"(\d+(?:\.\d+)?)% do CDI", asset_name)
                if match and 'CDI' in bench_data and bench_data['CDI'] is not None:
                    p, b = float(match.group(1)), bench_data['CDI']
                    bench_data[asset_name] = (1 + (b.pct_change().fillna(0) * (p/100))).cumprod() * 100

        data_frames = []
        if user_series is not None:
            user_series.name = 'Carteira'
            user_series.index = pd.to_datetime(user_series.index)
            data_frames.append(user_series)

        if 'SELIC' in bench_data:
            self.selic_series = bench_data['SELIC']
        
        nomes_para_exibir = []
        for item in benchmarks_to_use:
            if isinstance(item, str): nomes_para_exibir.append(item)
            elif isinstance(item, dict): nomes_para_exibir.append(item.get('nome'))

        for nome in nomes_para_exibir:
            if nome in bench_data and bench_data[nome] is not None:
                s = bench_data[nome]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s.index = pd.to_datetime(s.index)
                s = pd.to_numeric(s, errors='coerce')
                s.name = nome
                data_frames.append(s)

        if data_frames:
            df_raw = pd.concat(data_frames, axis=1).sort_index()
            if start_date and end_date:
                df_raw = df_raw.loc[start_date:end_date]
                
            self.df_combined = df_raw.ffill()
            if 'Carteira' in self.df_combined.columns:
                self.df_combined = self.df_combined.dropna(subset=['Carteira'])
            else:
                self.df_combined = self.df_combined.dropna(how='all')
                
            if not self.df_combined.empty:
                for col in self.df_combined.columns:
                    first_idx = self.df_combined[col].first_valid_index()
                    if first_idx is not None:
                        base_val = self.df_combined.loc[first_idx, col]
                        if base_val != 0:
                            self.df_combined[col] = (self.df_combined[col] / base_val) * 100
        else:
            self.logger.warning("Nenhum dado disponível para análise.")

    def export_csv(self, df, name):
        path = self._get_path("dados", f"{name}.csv")
        df.to_csv(path, sep=';', decimal=',')

    def plot_twr_evolution(self, title_suffix="", return_fig=False):
        if self.df_combined.empty: return (None, None) if return_fig else None
        
        df = self.df_combined
        last_values = df.iloc[-1].sort_values(ascending=False)
        cols_sorted = last_values.index

        fig, ax = plt.subplots(figsize=(12, 7))
        for col in cols_sorted:
            if col == 'Carteira':
                ax.plot(df.index, df[col], label=f"{col} ({df[col].iloc[-1]-100:.1f}%)", linewidth=3, color='blue', zorder=10)
            else:
                ax.plot(df.index, df[col], label=f"{col} ({df[col].iloc[-1]-100:.1f}%)", linewidth=1.5, alpha=0.7)

        ax.set_title(f"Evolução TWR (Base 100) - {title_suffix}", fontsize=14)
        ax.set_ylabel("Performance")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        if return_fig:
            return fig, df
            
        path = self._get_path("graficos", f"twr_evolucao_{title_suffix}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_drawdown(self, title_suffix="", return_fig=False):
        if self.df_combined.empty: return (None, None) if return_fig else None
        
        drawdown = calculate_drawdown(self.df_combined)

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in drawdown.columns:
            if col == 'Carteira':
                ax.plot(drawdown.index, drawdown[col], label=col, color='red', linewidth=2)
                ax.fill_between(drawdown.index, drawdown[col], 0, color='red', alpha=0.1)
            else:
                ax.plot(drawdown.index, drawdown[col], label=col, linestyle='--', alpha=0.6)

        ax.set_title(f"Drawdown (Queda Máxima) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        if return_fig: return fig, drawdown
        plt.savefig(self._get_path("graficos", f"drawdown_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def plot_risk_return_scatter(self, title_suffix="", return_fig=False):
        if self.df_combined.empty: return (None, None) if return_fig else None
        df = self.df_combined
        
        daily_ret = df.pct_change()
        volatility = daily_ret.std() * np.sqrt(252)
        days = (df.index[-1] - df.index[0]).days
        total_ret = (df.iloc[-1] / df.iloc[0])
        cagr = (total_ret ** (365.25 / days)) - 1 if days > 0 else 0
        
        metrics = pd.DataFrame({'Volatilidade': volatility, 'Retorno (CAGR)': cagr, 'Sharpe': (cagr - 0.10) / volatility}).dropna()

        fig, ax = plt.subplots(figsize=(10, 8))
        for name, row in metrics.iterrows():
            color, size, alpha = ('red', 150, 1.0) if name == 'Carteira' else ('blue', 80, 0.6)
            ax.scatter(row['Volatilidade'], row['Retorno (CAGR)'], s=size, c=color, alpha=alpha, edgecolors='black')
            ax.text(row['Volatilidade'], row['Retorno (CAGR)'], f"  {name}", fontsize=9, va='center')

        ax.set_title(f"Risco x Retorno - {title_suffix}", fontsize=14)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if return_fig: return fig, metrics
        plt.savefig(self._get_path("graficos", f"risco_retorno_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def plot_rolling_volatility(self, window=252, title_suffix="", return_fig=False):
        if self.df_combined.empty: return (None, None) if return_fig else None
        
        rolling_vol = calculate_rolling_volatility(self.df_combined, window)

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in rolling_vol.columns:
            if col == 'Carteira':
                ax.plot(rolling_vol.index, rolling_vol[col], label=col, color='red', linewidth=2, zorder=10)
            else:
                ax.plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=1.5, alpha=0.7)

        ax.set_title(f"Volatilidade Móvel ({window} dias) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        if return_fig: return fig, rolling_vol
        plt.savefig(self._get_path("graficos", f"volatilidade_movel_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def plot_rolling_sharpe(self, window=252, title_suffix="", return_fig=False):
        if self.df_combined.empty: return (None, None) if return_fig else None
        
        rolling_sharpe = calculate_rolling_sharpe(self.df_combined, self.selic_series, rf_constant=0.10, window=window)

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in rolling_sharpe.columns:
            if col == 'Carteira':
                ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, color='red', linewidth=2, zorder=10)
            else:
                ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, linewidth=1.5, alpha=0.7)

        ax.set_title(f"Sharpe Ratio Móvel ({window} dias) - {title_suffix}", fontsize=14)
        ax.grid(True)
        ax.legend()
        
        if return_fig: return fig, rolling_sharpe
        plt.savefig(self._get_path("graficos", f"sharpe_movel_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def plot_irr_evolution(self, title_suffix="", return_fig=False):
        if not hasattr(self, 'portfolio_df') or self.portfolio_df is None or self.portfolio_df.empty:
            return (None, None) if return_fig else None

        df = self.portfolio_df.sort_values('date')
        dates_to_calc = df.set_index('date').resample('ME').last().index
        if df['date'].iloc[-1] not in dates_to_calc:
            dates_to_calc = dates_to_calc.union([df['date'].iloc[-1]])
        dates_to_calc = dates_to_calc[dates_to_calc >= df['date'].iloc[0]]
        
        irr_history = []
        valid_dates = []
        
        all_dates = df['date'].values
        all_flows = -1 * df['fluxo'].values 
        all_markets = df['vlr_mercado'].values
        
        for target_date in dates_to_calc:
            mask = all_dates <= target_date
            if not np.any(mask): continue
            
            last_idx = np.where(mask)[0][-1]
            if all_markets[last_idx] == 0 and np.sum(np.abs(all_flows[mask])) == 0: continue

            calc_flows = np.append(all_flows[mask], all_markets[last_idx])
            calc_dates = np.append(all_dates[mask], all_dates[last_idx])
            
            try:
                res = calculate_xirr(calc_flows, pd.to_datetime(calc_dates))
                if res is not None and -0.99 < res < 10.0:
                    irr_history.append(res)
                    valid_dates.append(target_date)
            except Exception: pass

        if not irr_history: return (None, None) if return_fig else None

        series_irr = pd.Series(irr_history, index=valid_dates) * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series_irr.index, series_irr.values, label='TIR (Carteira)', color='purple', linewidth=2)
        ax.set_title(f"Evolução da TIR (Taxa Interna de Retorno) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True)
        ax.legend()
        
        if return_fig: return fig, series_irr
        plt.savefig(self._get_path("graficos", f"tir_evolucao_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def simulate_shadow_portfolios(self, title_suffix="", return_fig=False):
        if self.df_combined.empty or getattr(self, 'portfolio_df', None) is None:
            return (None, None) if return_fig else None

        df_flows = self.portfolio_df.set_index('date')[['fluxo', 'vlr_mercado', 'vlr_investido']]
        df_flows.index = pd.to_datetime(df_flows.index)
        
        df_prices = self.df_combined.copy()
        common_index = df_prices.index
        
        flows_aligned = df_flows['fluxo'].groupby(df_flows.index).sum().reindex(common_index, fill_value=0.0)
        shadow_wealth = pd.DataFrame(index=common_index)
        
        shadow_wealth['Carteira Real'] = df_flows['vlr_mercado'].reindex(common_index).ffill()
        shadow_wealth['Total Investido'] = flows_aligned.cumsum() + (df_flows['vlr_investido'].iloc[0] if not df_flows.empty else 0)

        for col in df_prices.columns:
            if col == 'Carteira': continue
            price_series = df_prices[col]
            shares_flow = flows_aligned.div(price_series).fillna(0)
            cum_shares = shares_flow.cumsum()
            initial_balance = df_flows['vlr_mercado'].iloc[0] if not df_flows.empty else 0
            initial_shares = initial_balance / price_series.iloc[0] if price_series.iloc[0] != 0 else 0
            shadow_wealth[col] = (cum_shares + initial_shares) * price_series

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(shadow_wealth.index, shadow_wealth['Total Investido'], color='gray', linestyle=':', label='Total Investido', linewidth=1.5)
        
        if 'Carteira Real' in shadow_wealth.columns:
            ax.plot(shadow_wealth.index, shadow_wealth['Carteira Real'], color='blue', linewidth=3, label='Carteira Real', zorder=10)

        for col in shadow_wealth.columns:
            if col not in ['Total Investido', 'Carteira Real']:
                ax.plot(shadow_wealth.index, shadow_wealth[col], linestyle='--', linewidth=1.5, label=col)

        ax.set_title(f"Simulação de Aportes: Carteira Real vs Benchmarks - {title_suffix}", fontsize=14)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if return_fig: return fig, shadow_wealth
        plt.savefig(self._get_path("graficos", f"simulacao_aportes_{title_suffix}.png"), bbox_inches='tight')
        plt.close()

    def generate_summary_table(self, title_suffix=""):
        if self.df_combined.empty: return
        df = self.df_combined
        
        total_ret = (df.iloc[-1] / df.iloc[0]) - 1
        yearly = df.resample('YE').last()
        yearly_ret = yearly.pct_change()
        yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1
        
        summary = yearly_ret.T
        summary.columns = [c.year for c in summary.columns]
        summary['Total Acum.'] = total_ret
        summary_fmt = summary.map(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
        self.export_csv(summary_fmt, f"resumo_rentabilidade_{title_suffix}")
