import os
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
from app.config import (
    BENCHMARKS_YF, BENCHMARKS_B3, BENCHMARKS_BCB, 
    BENCHMARKS_TD, CARTEIRAS_SINTETICAS, BENCHMARKS_EXIBIR
)

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
            return None

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
        return df_grp.set_index('date')['twr_index']

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

        # 2. Busca Benchmarks (Market Data)
        bench_data = processar_benchmarks(
            start_date, end_date, 
            BENCHMARKS_YF, BENCHMARKS_B3, BENCHMARKS_BCB, 
            BENCHMARKS_TD, CARTEIRAS_SINTETICAS, self.logger
        )

        # 3. Unificação
        data_frames = []
        
        # Adiciona Carteira (se houver)
        if user_series is not None:
            user_series.name = 'Carteira'
            data_frames.append(user_series)

        # Adiciona Benchmarks (apenas os configurados para exibir)
        # Mas guarda SELIC separada para cálculo de risco
        if 'SELIC' in bench_data:
            self.selic_series = bench_data['SELIC']
        
        for nome in BENCHMARKS_EXIBIR:
            if nome in bench_data and bench_data[nome] is not None:
                s = bench_data[nome]
                # Garante que é numérico
                s = pd.to_numeric(s, errors='coerce')
                s.name = nome
                data_frames.append(s)

        # Concatena tudo alinhando pelo índice (Data)
        if data_frames:
            self.df_combined = pd.concat(data_frames, axis=1).sort_index()
            # Preenche buracos (feriados locais vs globais) com o valor anterior
            self.df_combined = self.df_combined.ffill().dropna()
            
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

    def plot_twr_evolution(self, title_suffix=""):
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
        plt.savefig(self._get_path("graficos", f"twr_evolucao_{title_suffix}.png"), bbox_inches='tight')
        self.export_csv(df, f"twr_evolucao_{title_suffix}")
        plt.close()

    def plot_drawdown(self, title_suffix=""):
        """Calcula e plota o Drawdown (Queda do topo)."""
        if self.df_combined.empty: return

        # Cálculo do Drawdown: (Preço / Máximo_Acumulado) - 1
        rolling_max = self.df_combined.cummax()
        drawdown = (self.df_combined / rolling_max) - 1

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plota apenas Carteira e Top 3 Benchmarks para não poluir
        cols_to_plot = ['Carteira'] if 'Carteira' in drawdown.columns else []
        cols_to_plot += [c for c in drawdown.columns if c != 'Carteira'][:3]
        
        for col in cols_to_plot:
            if col not in drawdown.columns: continue
            
            if col == 'Carteira':
                ax.plot(drawdown.index, drawdown[col], label=col, color='red', linewidth=2)
                ax.fill_between(drawdown.index, drawdown[col], 0, color='red', alpha=0.1)
            else:
                ax.plot(drawdown.index, drawdown[col], label=col, linestyle='--', alpha=0.6)

        ax.set_title(f"Drawdown (Queda Máxima) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        plt.savefig(self._get_path("graficos", f"drawdown_{title_suffix}.png"), bbox_inches='tight')
        self.export_csv(drawdown, f"drawdown_{title_suffix}")
        plt.close()

    def plot_risk_return_scatter(self, title_suffix=""):
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

        plt.savefig(self._get_path("graficos", f"risco_retorno_{title_suffix}.png"), bbox_inches='tight')
        self.export_csv(metrics, f"metricas_risco_{title_suffix}")
        plt.close()

    def plot_rolling_volatility(self, window=252, title_suffix=""):
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
            elif col in BENCHMARKS_EXIBIR or len(rolling_vol.columns) <= 5:
                 ax.plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=1.5, alpha=0.7)

        ax.set_title(f"Volatilidade Móvel ({window} dias) - {title_suffix}", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        plt.savefig(self._get_path("graficos", f"volatilidade_movel_{title_suffix}.png"), bbox_inches='tight')
        self.export_csv(rolling_vol, f"volatilidade_movel_{title_suffix}")
        plt.close()

    def plot_rolling_sharpe(self, window=252, title_suffix=""):
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
        else:
             # Fallback: 10% a.a. convertido para diário
             rf_daily_series[:] = (1.10 ** (1/252)) - 1

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
            elif col in BENCHMARKS_EXIBIR or len(rolling_sharpe.columns) <= 5:
                ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, linewidth=1.5, alpha=0.7)

        ax.set_title(f"Sharpe Ratio Móvel ({window} dias) - {title_suffix}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.axhline(0, color='black', linewidth=1)

        plt.savefig(self._get_path("graficos", f"sharpe_movel_{title_suffix}.png"), bbox_inches='tight')
        self.export_csv(rolling_sharpe, f"sharpe_movel_{title_suffix}")
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
        summary_fmt = summary.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
        
        self.export_csv(summary_fmt, f"resumo_rentabilidade_{title_suffix}")


def main():
    parser = argparse.ArgumentParser(description="Gera relatórios financeiros consolidados (V2).")
    parser.add_argument('--debug', action='store_true', help='Log detalhado.')
    
    # Modos de Operação
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--historico', type=int, help='Anos de histórico para análise de mercado (sem carteira).')
    group.add_argument('--ativo', type=str, help='Código do ativo na carteira do usuário.')
    group.add_argument('--classe', type=str, help='Classe de ativos na carteira do usuário.')
    
    args = parser.parse_args()

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
    
    # Tabela Resumo
    report.generate_summary_table(title_suffix=nome_analise)

    logger.info("Processo concluído com sucesso (V2).")

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