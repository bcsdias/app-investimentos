# App de Investimentos

Ferramenta para análise de carteira, cálculo de TWR (Time-Weighted Return), comparação com benchmarks e simulação de aportes.

## Estrutura do Projeto

- **app/**: Código principal (`main.py`).
- **utils/**: Módulos auxiliares (logger, importação B3).
- **data/**: Dados brutos e downloads.
- **reports/**: Gráficos e relatórios gerados.

## Configuração do Ambiente

1. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   ```

2. **Ative o ambiente virtual:**
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Como Rodar

Certifique-se de que o arquivo `.env` com o `DLP_TOKEN` esteja na raiz do projeto.

### Exemplos de uso:

**1. Histórico de 10 anos com simulação de aportes:**
```bash
python app/main.py --historico 10 --aporte 1000 --rebalanceamento 6
```

**2. Analisar um ativo específico:**
```bash
python app/main.py --ativo KLBN11
```