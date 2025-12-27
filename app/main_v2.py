import os
import time
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv

# Configuração de Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.logger import setup_logger
from utils.market_data import processar_benchmarks
from app.benchmarks_config import CATALOGO_YF, CATALOGO_B3, CATALOGO_BCB, CATALOGO_TD, BENCHMARKS_ATIVOS

# Carrega ambiente
load_dotenv()

class FinancialReport:
    def __init__(self, logger, output_dir="reports"):
        self.logger = logger
        self.base_output_dir = os.path.join(BASE_DIR, output_dir)
        self.df_combined = pd.DataFrame() # DataFrame Mestre (Carteira + Benchmarks)
        self.risk_free_rate = 0.0 # Será preenchido com a SELIC média ou atual
        self.selic_series = None

    def _get_path(self, subfolder, filename):
        """Gera caminho completo e cria pasta se não existir."""
        folder = os.path.join(self.base_output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def fetch_user_portfolio(self, token, ativo=None, classe=None):
        """Busca dados da API e calcula o TWR da carteira."""
        from utils.market_data import buscar_historico # Import local para evitar ciclo se houver
        
        self.logger.info("Buscando histórico da carteira do usuário...")
        df = buscar_historico(token, self.logger, ativo=ativo, classe=classe)
        
        if df is None or df.empty:
            self.logger.warning("Nenhum dado retornado pela API ou DataFrame vazio.")
            return None
        self.logger.debug(f"Dados brutos recebidos da API: {df.shape[0]} linhas. Colunas: {list(df.columns)}")

        # --- Cálculo do TWR (Simplificado e Extraído) ---
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Agrupa por data (caso haja múltiplos ativos na mesma classe)
        df_grp = df.groupby('date')[['vlr_mercado', 'vlr_investido', 'proventos']].sum().reset_index()
        
        # Lógica TWR
        df_grp['fluxo'] = df_grp['vlr_investido'].diff().fillna(df_grp['vlr_investido'].iloc[0]) - df_grp['proventos']
        df_grp['vlr_inicial'] = df_grp['vlr_mercado'].shift(1).fillna(0)
        
        # HPR (Holding Period Return)
        denominador = df_grp['vlr_inicial'] + df_grp['fluxo']
        df_grp['hpr'] = np.where(denominador != 0, df_grp['vlr_mercado'] / denominador, 1.0)
        
        # Tratamento para primeiro aporte ou zeragem
        mask_zeros = (df_grp['vlr_mercado'] == 0) & (df_grp['vlr_inicial'] == 0)
        df_grp.loc[mask_zeros, 'hpr'] = 1.0
        
        # TWR Acumulado (Base 1.0 para facilitar comparação com benchmarks)
        df_grp['twr_index'] = df_grp['hpr'].cumprod()
        
        # Retorna Série indexada por data
        result_series = df_grp.set_index('date')['twr_index']
        self.logger.info(f"Carteira processada: {len(result_series)} dias de histórico ({result_series.index.min().date()} a {result_series.index.max().date()}).")
        self.logger.debug(f"Amostra TWR Carteira (Head):\n{result_series.head().to_string()}")
        
        # Armazena DataFrame processado para cálculo de TIR (fluxos)
        self.portfolio_df = df_grp.copy()
        return result_series

    def build_dataset(self, user_series=None, years_history=None):
        """
        Constrói o DataFrame unificado (Carteira + Benchmarks).
        Se user_series existir, usa as datas dela. Se não, usa years_history.
        """
        # 1. Definição de Datas
        if user_series is not None:
            start_date = user_series.index.min().strftime('%Y-%m-%d')
            end_date = user_series.index.max().strftime('%Y-%m-%d')
            self.logger.info(f"Período definido pela carteira: {start_date} a {end_date}")
        else:
            end_dt = pd.Timestamp.today()
            start_dt = end_dt - pd.DateOffset(years=years_history)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            self.logger.info(f"Período definido por histórico ({years_history} anos): {start_date} a {end_date}")

        # 2. Processa Configuração e Filtra Catálogos
        # Identifica quais ativos base precisam ser baixados com base na configuração ativa
        needed_assets = set()
        carteiras_sinteticas = {}
        
        # Sempre tenta baixar SELIC para cálculo de risco (se disponível no catálogo)
        if 'SELIC' in CATALOGO_BCB:
            needed_assets.add('SELIC')

        for item in BENCHMARKS_ATIVOS:
            if isinstance(item, str):
                needed_assets.add(item)
            elif isinstance(item, dict):
                nome = item.get('nome')
                comps = item.get('composicao')
                if nome and comps:
                    carteiras_sinteticas[nome] = comps
                    # Adiciona componentes da carteira à lista de necessários
                    for comp_name in comps.keys():
                        needed_assets.add(comp_name)

        # Resolve dependências implícitas (ex: 'IMID BRL' precisa de 'IMID')
        final_needed = set(needed_assets)
        for asset in needed_assets:
            if isinstance(asset, str) and asset.endswith(' BRL'):
                final_needed.add(asset.replace(' BRL', ''))
            if asset == 'IPCA + 6%':
                final_needed.add('IPCA')

        # Filtra os catálogos para baixar apenas o necessário
        yf_filtered = {k: v for k, v in CATALOGO_YF.items() if k in final_needed}
        b3_filtered = {k: v for k, v in CATALOGO_B3.items() if k in final_needed}
        bcb_filtered = {k: v for k, v in CATALOGO_BCB.items() if k in final_needed}
        td_filtered = {k: v for k, v in CATALOGO_TD.items() if k in final_needed}

        # 3. Busca Benchmarks (Market Data)
        self.logger.info(f"Solicitando dados de mercado. Fontes filtradas: YF={len(yf_filtered)}, B3={len(b3_filtered)}, BCB={len(bcb_filtered)}, TD={len(td_filtered)}")
        bench_data = processar_benchmarks(
            start_date, end_date,
            yf_filtered, b3_filtered, bcb_filtered,
            td_filtered, carteiras_sinteticas, self.logger
        )
        
        loaded_benchmarks = [k for k, v in bench_data.items() if v is not None and not v.empty]
        self.logger.debug(f"Benchmarks carregados com sucesso ({len(loaded_benchmarks)}): {loaded_benchmarks}")

        # 4. Unificação
        data_frames = []
        
        # Adiciona Carteira (se houver)
        if user_series is not None:
            user_series.name = 'Carteira'
            data_frames.append(user_series)

        # Adiciona Benchmarks (apenas os configurados para exibir)
        # Mas guarda SELIC separada para cálculo de risco
        if 'SELIC' in bench_data:
            self.selic_series = bench_data['SELIC']
        
        # Adiciona apenas os itens listados explicitamente em BENCHMARKS_ATIVOS
        # (Isso filtra ativos base que foram baixados apenas como dependência, ex: 'IMID' puro)
        nomes_para_exibir = []
        for item in BENCHMARKS_ATIVOS:
            if isinstance(item, str): nomes_para_exibir.append(item)
            elif isinstance(item, dict): nomes_para_exibir.append(item.get('nome'))

        for nome in nomes_para_exibir:
            if nome in bench_data and bench_data[nome] is not None:
                s = bench_data[nome]
                s = pd.to_numeric(s, errors='coerce')
                s.name = nome
                data_frames.append(s)

        # Concatena tudo alinhando pelo índice (Data)
        if data_frames:
            # Cria temporário para análise de qualidade dos dados
            df_raw = pd.concat(data_frames, axis=1).sort_index()
            
            # Log de diagnóstico de NaNs (Debug)
            nans = df_raw.isna().sum()
            if nans.sum() > 0:
                self.logger.debug(f"Valores ausentes (NaN) antes da limpeza:\n{nans[nans > 0].to_string()}")

            # Preenche buracos (feriados locais vs globais) com o valor anterior
            self.df_combined = df_raw.ffill().dropna()
            
            rows_dropped = len(df_raw) - len(self.df_combined)
            if rows_dropped > 0:
                self.logger.warning(f"Foram removidas {rows_dropped} linhas (dias) devido à falta de interseção de dados entre os ativos.")
            
            self.logger.info(f"Dataset consolidado: {self.df_combined.shape[0]} linhas x {self.df_combined.shape[1]} colunas. Período comum: {self.df_combined.index.min().date()} a {self.df_combined.index.max().date()}")
            
            # Normaliza tudo para Base 100 no início do período comum
            if not self.df_combined.empty:
                self.df_combined = (self.df_combined / self.df_combined.iloc[0]) * 100
        else:
            self.logger.warning("Nenhum dado disponível para análise.")

    def export_csv(self, df, name):
        """Salva DataFrame em CSV formatado."""
        path = self._get_path("dados", f"{name}.csv")
        df.to_csv(path, sep=';', decimal=',')
        self.logger.info(f"CSV salvo: {path}")

    # ==========================================
    # MÉTODOS DE ANÁLISE E PLOTAGEM
    # ==========================================

    def plot_twr_evolution(self, title_suffix="", return_fig=False):
        """Gera gráfico de linha comparativo (TWR)."""
        if self.df_combined.empty: return

        df = self.df_combined
        
        # Ordena legenda pela rentabilidade final
        last_values = df.iloc[-1].sort_values(ascending=False)
        cols_sorted = last_values.index

        fig, ax = plt.subplots(figsize=(12, 7))
        
        for col in cols_sorted:
            # Destaque para a Carteira
            if col == 'Carteira':
                ax.plot(df.index, df[col], label=f"{col} ({df[col].iloc[-1]-100:.1f}%)", linewidth=3, color='blue', zorder=10)
            else:
                ax.plot(df.index, df[col], label=f"{col} ({df[col].iloc[-1]-100:.1f}%)", linewidth=1.5, alpha=0.7)

        ax.set_title(f"Evolução TWR (Base 100) - {title_suffix}", fontsize=14)
        ax.set_ylabel("Performance")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Salva
        path = self._get_path("graficos", f"twr_evolucao_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico TWR: {path}")
        self.export_csv(df, f"twr_evolucao_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_drawdown(self, title_suffix="", return_fig=False):
        """Calcula e plota o Drawdown (Queda do topo)."""
        if self.df_combined.empty: return

        # Cálculo do Drawdown: (Preço / Máximo_Acumulado) - 1
        rolling_max = self.df_combined.cummax()
        drawdown = (self.df_combined / rolling_max) - 1

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

        path = self._get_path("graficos", f"drawdown_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico Drawdown: {path}")
        self.export_csv(drawdown, f"drawdown_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_risk_return_scatter(self, title_suffix="", return_fig=False):
        """Gera gráfico de dispersão Risco (Volatilidade) x Retorno (CAGR)."""
        if self.df_combined.empty: return

        df = self.df_combined
        
        # Retornos diários
        daily_ret = df.pct_change().dropna()
        
        # Métricas Anualizadas (252 dias úteis)
        volatility = daily_ret.std() * np.sqrt(252)
        
        # CAGR (Compound Annual Growth Rate)
        days = (df.index[-1] - df.index[0]).days
        total_ret = (df.iloc[-1] / df.iloc[0])
        cagr = (total_ret ** (365.25 / days)) - 1

        # Sharpe Ratio (Simplificado, assumindo RF constante se não tiver série)
        # Se tivermos a série SELIC alinhada, poderíamos fazer o cálculo exato.
        # Aqui faremos (CAGR - 10%) / Vol para simplificar a visualização ou usar a média da SELIC se disponível.
        rf = 0.10 # 10% a.a. default
        sharpe = (cagr - rf) / volatility

        # DataFrame de Métricas
        metrics = pd.DataFrame({
            'Volatilidade': volatility,
            'Retorno (CAGR)': cagr,
            'Sharpe': sharpe
        })

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, row in metrics.iterrows():
            color = 'red' if name == 'Carteira' else 'blue'
            size = 150 if name == 'Carteira' else 80
            alpha = 1.0 if name == 'Carteira' else 0.6
            
            ax.scatter(row['Volatilidade'], row['Retorno (CAGR)'], s=size, c=color, alpha=alpha, edgecolors='black')
            ax.text(row['Volatilidade'], row['Retorno (CAGR)'], f"  {name}", fontsize=9, va='center')

        ax.set_title(f"Risco x Retorno - {title_suffix}", fontsize=14)
        ax.set_xlabel("Risco (Volatilidade Anualizada)")
        ax.set_ylabel("Retorno Anualizado (CAGR)")
        
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Linhas de quadrante (média)
        ax.axhline(metrics['Retorno (CAGR)'].mean(), color='gray', linestyle=':', alpha=0.5)
        ax.axvline(metrics['Volatilidade'].mean(), color='gray', linestyle=':', alpha=0.5)

        path = self._get_path("graficos", f"risco_retorno_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico Risco x Retorno: {path}")
        self.export_csv(metrics, f"metricas_risco_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_rolling_volatility(self, window=252, title_suffix="", return_fig=False):
        """Gera gráfico de Volatilidade Móvel (anualizada)."""
        if self.df_combined.empty: return
        
        # Retornos diários
        daily_ret = self.df_combined.pct_change().dropna()
        
        # Volatilidade Móvel Anualizada (Janela de 'window' dias)
        rolling_vol = daily_ret.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

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

        path = self._get_path("graficos", f"volatilidade_movel_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico Volatilidade Móvel: {path}")
        self.export_csv(rolling_vol, f"volatilidade_movel_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_rolling_sharpe(self, window=252, title_suffix="", return_fig=False):
        """Gera gráfico de Sharpe Ratio Móvel."""
        if self.df_combined.empty: return
        
        daily_ret = self.df_combined.pct_change().dropna()
        
        # Define Taxa Livre de Risco (Diária)
        rf_daily_series = pd.Series(0.0, index=daily_ret.index)
        
        if self.selic_series is not None:
             # Calcula taxa diária a partir do índice acumulado da SELIC
             selic_daily = self.selic_series.pct_change().fillna(0)
             # Alinha com as datas do dataframe
             rf_daily_series = selic_daily.reindex(daily_ret.index).ffill().fillna(0)
             self.logger.debug("Usando série histórica da SELIC para cálculo do Sharpe.")
        else:
             # Fallback: 10% a.a. convertido para diário
             rf_daily_series[:] = (1.10 ** (1/252)) - 1
             self.logger.info("Série SELIC não disponível. Usando taxa fixa de 10% a.a. como Risk Free para o Sharpe.")

        # Excesso de retorno (Retorno Ativo - Risk Free)
        excess_ret = daily_ret.sub(rf_daily_series, axis=0)
        
        # Média e Volatilidade Móveis
        rolling_mean = excess_ret.rolling(window=window).mean()
        rolling_std = excess_ret.rolling(window=window).std()
        
        # Sharpe Anualizado = (Média Diária / Vol Diária) * sqrt(252)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()

        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in rolling_sharpe.columns:
            if col == 'Carteira':
                ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, color='red', linewidth=2, zorder=10)
            else:
                ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, linewidth=1.5, alpha=0.7)

        ax.set_title(f"Sharpe Ratio Móvel ({window} dias) - {title_suffix}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.axhline(0, color='black', linewidth=1)

        path = self._get_path("graficos", f"sharpe_movel_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico Sharpe Móvel: {path}")
        self.export_csv(rolling_sharpe, f"sharpe_movel_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def _calculate_xirr(self, cash_flows, dates):
        """Calcula a TIR (XIRR) usando Newton-Raphson."""
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

    def plot_irr_evolution(self, title_suffix="", return_fig=False):
        """Gera gráfico da evolução da TIR (Taxa Interna de Retorno)."""
        if not hasattr(self, 'portfolio_df') or self.portfolio_df is None or self.portfolio_df.empty:
            return

        df = self.portfolio_df.sort_values('date')
        
        # Amostragem mensal para performance (calcular dia a dia é muito pesado)
        dates_to_calc = df.set_index('date').resample('ME').last().index
        # Garante inclusão da última data real
        if df['date'].iloc[-1] not in dates_to_calc:
            dates_to_calc = dates_to_calc.union([df['date'].iloc[-1]])
        dates_to_calc = dates_to_calc[dates_to_calc >= df['date'].iloc[0]]
        
        irr_history = []
        valid_dates = []
        
        all_dates = df['date'].values
        # Fluxo para TIR: Investimento é negativo (-fluxo), Resgate é positivo
        # Na nossa lógica: fluxo = investido.diff - proventos. 
        # Se investi 1000, fluxo=1000. Para TIR deve ser -1000.
        all_flows = -1 * df['fluxo'].values 
        all_markets = df['vlr_mercado'].values
        
        for target_date in dates_to_calc:
            # Filtra dados até a data alvo
            mask = all_dates <= target_date
            if not np.any(mask): continue
            
            current_flows = all_flows[mask]
            current_dates = all_dates[mask]
            
            # Adiciona Valor de Mercado atual como fluxo positivo (resgate fictício)
            last_idx = np.where(mask)[0][-1]
            current_market_val = all_markets[last_idx]
            
            # Se valor de mercado é zero e não há fluxos relevantes, pula
            if current_market_val == 0 and np.sum(np.abs(current_flows)) == 0: continue

            # Monta arrays finais para cálculo
            calc_flows = np.append(current_flows, current_market_val)
            calc_dates = np.append(current_dates, all_dates[last_idx])
            
            try:
                res = self._calculate_xirr(calc_flows, pd.to_datetime(calc_dates))
                if res is not None and -0.99 < res < 10.0: # Filtra outliers extremos
                    irr_history.append(res)
                    valid_dates.append(target_date)
            except Exception:
                pass

        if not irr_history:
            return

        series_irr = pd.Series(irr_history, index=valid_dates) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series_irr.index, series_irr.values, label='TIR (Carteira)', color='purple', linewidth=2)
        
        ax.set_title(f"Evolução da TIR (Taxa Interna de Retorno) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.axhline(0, color='black', linewidth=1)

        path = self._get_path("graficos", f"tir_evolucao_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico TIR: {path}")
        self.export_csv(series_irr, f"tir_evolucao_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def simulate_shadow_portfolios(self, title_suffix="", return_fig=False):
        """
        Simula o patrimônio se os mesmos aportes/resgates da carteira tivessem sido
        feitos nos benchmarks (Shadow Portfolio).
        """
        if self.df_combined.empty or not hasattr(self, 'portfolio_df') or self.portfolio_df is None:
            return

        # 1. Prepara os dados de Fluxo da Carteira Real
        # Garante que o índice é datetime
        df_flows = self.portfolio_df.set_index('date')[['fluxo', 'vlr_mercado', 'vlr_investido']].copy()
        df_flows.index = pd.to_datetime(df_flows.index)
        
        # 2. Prepara os Benchmarks (Preços/Índices)
        # Usa o df_combined que já tem os benchmarks alinhados e limpos
        # (Não importa que esteja em base 100, pois a variação relativa é o que conta)
        df_prices = self.df_combined.copy()
        
        # 3. Alinha Fluxos com Preços (Reindexa para garantir mesmas datas)
        # Mantém apenas datas onde temos preços de benchmark (intersecção)
        common_index = df_prices.index
        
        # Cria série de fluxos alinhada (preenche dias sem aporte com 0)
        # Agrupa fluxos por dia (caso haja duplicidade) e reindexa
        flows_aligned = df_flows['fluxo'].groupby(df_flows.index).sum().reindex(common_index, fill_value=0.0)
        
        # DataFrame para guardar os resultados (Patrimônio em R$)
        shadow_wealth = pd.DataFrame(index=common_index)
        
        # Adiciona a Carteira Real (Valor de Mercado original)
        # Reindexa e preenche buracos (forward fill para dias sem cotação na carteira mas com cotação no mercado)
        shadow_wealth['Carteira Real'] = df_flows['vlr_mercado'].reindex(common_index).ffill()
        
        # Adiciona linha de "Total Investido" (Acumulado dos fluxos)
        shadow_wealth['Total Investido'] = flows_aligned.cumsum() + (df_flows['vlr_investido'].iloc[0] if not df_flows.empty else 0)

        # 4. Simulação para cada Benchmark
        for col in df_prices.columns:
            if col == 'Carteira': continue # Pula a própria carteira (já tratada acima)
            
            price_series = df_prices[col]
            
            # Quantidade de cotas compradas/vendidas = Fluxo / Preço do Dia
            # Se preço for 0 ou NaN, não compra nada
            shares_flow = flows_aligned.div(price_series).fillna(0)
            
            # Acumula quantidade de cotas (Posição Custódia)
            cum_shares = shares_flow.cumsum()
            
            # Valor Patrimonial = Cotas Acumuladas * Preço Atual
            # Adiciona valor inicial investido (se houver saldo inicial na carteira real antes do periodo)
            initial_balance = df_flows['vlr_mercado'].iloc[0] if not df_flows.empty else 0
            # Ajuste simples: assume que o saldo inicial compraria cotas no dia 0
            initial_shares = initial_balance / price_series.iloc[0]
            
            shadow_wealth[col] = (cum_shares + initial_shares) * price_series

        # 5. Plotagem
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plota Total Investido (Referência)
        final_investido = shadow_wealth['Total Investido'].iloc[-1]
        ax.plot(shadow_wealth.index, shadow_wealth['Total Investido'], label=f'Total Investido (Caixa) (R$ {final_investido:,.0f})', 
                color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Plota Carteira Real
        if 'Carteira Real' in shadow_wealth.columns:
            final_real = shadow_wealth['Carteira Real'].iloc[-1]
            ax.plot(shadow_wealth.index, shadow_wealth['Carteira Real'], label=f'Carteira Real (R$ {final_real:,.0f})', 
                    color='blue', linewidth=3, zorder=10)
            # Preenche área da carteira real
            ax.fill_between(shadow_wealth.index, shadow_wealth['Carteira Real'], 0, color='blue', alpha=0.05)

        # Plota Benchmarks Simulados
        for col in shadow_wealth.columns:
            if col in ['Total Investido', 'Carteira Real']: continue
            
            # Pega o valor final para a legenda
            final_val = shadow_wealth[col].iloc[-1]
            ax.plot(shadow_wealth.index, shadow_wealth[col], label=f"{col} (R$ {final_val:,.0f})", 
                    linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_title(f"Simulação de Aportes: Carteira Real vs Benchmarks - {title_suffix}", fontsize=14)
        ax.set_ylabel("Patrimônio (R$)")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('R${x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        path = self._get_path("graficos", f"simulacao_aportes_{title_suffix}.png")
        self.logger.info(f"Gerando gráfico Simulação de Aportes: {path}")
        self.export_csv(shadow_wealth, f"simulacao_aportes_{title_suffix}")
        
        if return_fig:
            return fig
            
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def generate_summary_table(self, title_suffix=""):
        """Gera tabela resumo com Rentabilidade Total, Ano a Ano e Volatilidade."""
        if self.df_combined.empty: return
        
        df = self.df_combined
        
        # Rentabilidade Total
        total_ret = (df.iloc[-1] / df.iloc[0]) - 1
        
        # Rentabilidade Anual
        # Resample anual pegando o último valor
        yearly = df.resample('YE').last()
        yearly_ret = yearly.pct_change()
        # Ajuste do primeiro ano
        first_year_ret = (yearly.iloc[0] / df.iloc[0]) - 1
        yearly_ret.iloc[0] = first_year_ret
        
        # Transpõe para formato Tabela (Linhas=Ativos, Colunas=Anos)
        summary = yearly_ret.T
        summary.columns = [c.year for c in summary.columns]
        
        summary['Total Acum.'] = total_ret
        
        # Formatação (apenas para CSV visual, mantemos float para cálculo se precisar)
        summary_fmt = summary.map(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
        
        self.export_csv(summary_fmt, f"resumo_rentabilidade_{title_suffix}")


def main():
    parser = argparse.ArgumentParser(description="Gera relatórios financeiros consolidados (V2).")
    parser.add_argument('--debug', action='store_true', help='Log detalhado.')
    
    # Modos de Operação
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--historico', type=int, help='Anos de histórico para análise de mercado (sem carteira).')
    group.add_argument('--ativo', type=str, help='Código do ativo na carteira do usuário.')
    group.add_argument('--classe', type=str, help='Classe de ativos na carteira do usuário.')
    parser.add_argument('--simular-aportes', action='store_true', help='Simula o desempenho se os aportes fossem feitos nos benchmarks.')
    
    args = parser.parse_args()

    start_time = time.time()
    # Setup
    logger = setup_logger(debug=args.debug, log_file='main_v2.log')
    token = os.getenv('DLP_TOKEN')
    
    if not token and (args.ativo or args.classe):
        logger.error("Token DLP_TOKEN não encontrado para buscar dados da carteira.")
        return

    report = FinancialReport(logger)
    
    # 1. Obtenção de Dados da Carteira (Se aplicável)
    user_series = None
    nome_analise = ""
    
    if args.ativo or args.classe:
        nome_analise = args.ativo if args.ativo else args.classe
        logger.info(f"Iniciando análise de carteira: {nome_analise}")
        
        user_series = report.fetch_user_portfolio(token, ativo=args.ativo, classe=args.classe)
        
        if user_series is None:
            logger.error("Não foi possível obter dados da carteira. Encerrando.")
            return
    else:
        nome_analise = f"Mercado_{args.historico}anos"
        logger.info(f"Iniciando análise de mercado (Standalone): {args.historico} anos")

    # 2. Construção do Dataset Unificado (Carteira + Benchmarks)
    # Se user_series for None, ele usa args.historico para definir as datas
    report.build_dataset(user_series=user_series, years_history=args.historico)

    # 3. Geração de Artefatos (Gráficos e CSVs)
    logger.info("Gerando gráficos e relatórios...")
    
    # TWR (Evolução)
    report.plot_twr_evolution(title_suffix=nome_analise)
    
    # Drawdown
    report.plot_drawdown(title_suffix=nome_analise)
    
    # Risco x Retorno (Sharpe implícito)
    report.plot_risk_return_scatter(title_suffix=nome_analise)
    
    # Volatilidade Móvel (Evolução do Risco)
    report.plot_rolling_volatility(title_suffix=nome_analise)

    # Sharpe Móvel (Evolução da Eficiência)
    report.plot_rolling_sharpe(title_suffix=nome_analise)
    
    # TIR (Evolução da Rentabilidade Real)
    report.plot_irr_evolution(title_suffix=nome_analise)
    
    # Simulação de Aportes (Shadow Portfolio)
    if args.simular_aportes and (args.ativo or args.classe):
        report.simulate_shadow_portfolios(title_suffix=nome_analise)
    
    # Tabela Resumo
    report.generate_summary_table(title_suffix=nome_analise)

    elapsed_time = time.time() - start_time
    logger.info(f"Processo concluído com sucesso (V2). Tempo total: {elapsed_time:.2f}s")

if __name__ == "__main__":
    main()
'''

### Principais Otimizações Realizadas:

1.  **Classe `FinancialReport`:** Centraliza o estado (dados, diretórios, logger). Isso elimina a necessidade de passar 5-6 argumentos para cada função (como acontecia no `main.py` antigo).
2.  **DataFrame Unificado (`df_combined`):**
   *   Em vez de tratar a carteira e os benchmarks separadamente, o script agora cria um único DataFrame onde a coluna `Carteira` (se existir) é tratada matematicamente igual às colunas `IBOV`, `CDI`, etc.
   *   Isso permite que funções como `plot_drawdown` ou `plot_risk_return_scatter` sejam genéricas. Elas funcionam se você passar só benchmarks (modo `--historico`) ou benchmarks + carteira (modo `--ativo`).
3.  **Cálculo de TWR Isolado:** A lógica de cálculo do TWR (Time-Weighted Return) foi extraída para `fetch_user_portfolio`. Ela retorna uma Série limpa e indexada por data, pronta para ser mesclada com os dados de mercado.
4.  **Geração Automática de CSV:** Cada função de plotagem (`plot_*`) chama `self.export_csv` ao final, garantindo que para cada imagem gerada, existe um CSV correspondente com os dados brutos, conforme solicitado.
5.  **Métricas Solicitadas:**
   *   **TWR:** Gráfico de linha (`plot_twr_evolution`).
   *   **Drawdown:** Gráfico de área/linha (`plot_drawdown`).
   *   **Volatilidade e Sharpe:** Gráfico de dispersão (`plot_risk_return_scatter`) e tabela CSV (`metrics`).
   *   **TIR:** Embora a TIR exata exija fluxos de caixa precisos, o TWR é a métrica padrão da indústria para comparação gráfica. O script foca no TWR para os gráficos e calcula o CAGR (Retorno Anualizado) para a tabela de risco, que serve como proxy de rentabilidade para comparação.

Para rodar este novo script, você usaria comandos similares:
*   **Apenas Mercado:** `python app/main_v2.py --historico 5`
*   **Carteira vs Mercado:** `python app/main_v2.py --classe AÇÃO`

<!--
[PROMPT_SUGGESTION]Poderia adicionar no main_v2.py uma função para calcular a Matriz de Correlação entre a carteira e os benchmarks e salvar como um mapa de calor (heatmap)?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Como eu poderia adaptar o método fetch_user_portfolio para aceitar um arquivo CSV local de transações em vez de chamar a API, para fins de teste offline?[/PROMPT_SUGGESTION]
'''