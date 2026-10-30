"""Payload /resumo e corpo de /plan de teste. Saldos: BTC 30000 · CDB 12000 (custo) · PETR4 11430 · HGLG11 8000 · IMAB11 5000."""

RESUMO = {"wallet": [
    {"ativo": "PETR4", "classe": "ACAO", "qtd": 300, "price_fx": 38.10, "pm_fx": 30.00,
     "vlr_investido": 9000.0, "market_cod": "BVMF:PETR4", "moeda": "BRL"},
    {"ativo": "HGLG11", "classe": "FII", "qtd": 50, "price_fx": 160.0, "pm_fx": 150.0,
     "vlr_investido": 7500.0, "market_cod": "BVMF:HGLG11", "moeda": "BRL"},
    {"ativo": "CDB-XP-2027", "classe": "RF", "qtd": 1, "price_fx": None, "pm_fx": None,
     "vlr_investido": 12000.0, "market_cod": "", "moeda": "BRL"},
    {"ativo": "IMAB11", "classe": "ETF RF", "qtd": 50, "price_fx": 100.0, "pm_fx": 90.0,
     "vlr_investido": 4500.0, "market_cod": "BVMF:IMAB11", "moeda": "BRL"},
    {"ativo": "BTC", "classe": "CRIPTO", "qtd": 0.1, "price_fx": 300000.0, "pm_fx": 200000.0,
     "vlr_investido": 20000.0, "market_cod": "CURRENCY:BTC", "moeda": "BRL"},
]}

PLAN_BODY = {
    "targets": [{"ticker": "IMAB11", "weight": 60}, {"ticker": "VWRA11", "weight": 35}, {"ticker": "BITH11", "weight": 5}],
    "holdings": [{"ativo": "PETR4", "action": "sell"}, {"ativo": "HGLG11", "action": "sell"},
                 {"ativo": "CDB-XP-2027", "action": "count_as", "target": "IMAB11", "saldo": 12850.0}],
}

ANA_RESUMO = {"wallet": [{"ativo": "VALE3", "classe": "ACAO", "qtd": 10, "price_fx": 60.0, "pm_fx": 50.0,
                          "vlr_investido": 500.0, "market_cod": "BVMF:VALE3", "moeda": "BRL"}]}
