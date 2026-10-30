"""PDF do plano (fpdf2). Porta do layout de gerar_pdf_plano do legado, com alvos dinâmicos e IR.

Renderiza o envelope que /plan devolve — nada é recalculado aqui.
"""
from __future__ import annotations

import datetime as dt

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from apps.migration_ir.schemas import PlanRequest
from apps.migration_ir.tax import DISCLAIMER

_NAVY = (26, 54, 93)
_LINE = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}


def _t(text) -> str:
    """Fonte core helvetica só aceita latin-1."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _brl(v: float | None) -> str:
    if v is None:
        return "-"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def _qtd(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0").rstrip(".").replace(".", ",")


class _PlanPDF(FPDF):
    def __init__(self, generated_at: dt.datetime):
        super().__init__()
        self.generated_at = generated_at

    def header(self):
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 22, "F")
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.set_font("helvetica", "B", 13)
        self.cell(0, 5, _t("PLANO DE MIGRAÇÃO MENSAL E ISENÇÃO DE IR"), align="C", **_LINE)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 5, _t("Simulação de transição patrimonial"), align="C", **_LINE)
        self.set_text_color(0, 0, 0)
        self.set_y(26)

    def footer(self):
        self.set_y(-18)
        self.set_font("helvetica", "I", 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 4, _t(DISCLAIMER), **_LINE)
        stamp = self.generated_at.strftime("%d/%m/%Y %H:%M")
        self.cell(100, 6, _t(f"Gerado em {stamp} | app-investimentos"), align="L")
        self.cell(90, 6, f"Página {self.page_no()}/{{nb}}", align="R")

    def section(self, title: str):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*_NAVY)
        self.cell(0, 7, _t(title), **_LINE)
        self.set_draw_color(*_NAVY)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(50, 50, 50)


def _parameters(pdf: _PlanPDF, params: PlanRequest, meta: dict):
    pdf.section("1. Parâmetros da Estratégia")
    pdf.set_font("helvetica", "", 9)
    if params.contributions_only:
        pdf.cell(0, 5, _t("Modo: Apenas Aportes (sem vendas)"), **_LINE)
        pdf.cell(0, 5, _t(f"Total a aportar: {_brl(params.initial_cash)}"), **_LINE)
    else:
        pdf.cell(95, 5, _t(f"Teto de venda de ações por mês: {_brl(meta['monthly_cap'])}"))
        pdf.cell(95, 5, _t(f"Saldo em caixa: {_brl(params.initial_cash)}"), **_LINE)
        pdf.cell(0, 5, _t(f"Vendas de ações já feitas no mês: {_brl(params.already_sold_this_month)}"), **_LINE)
    pesos = " | ".join(f"{t.ticker.strip().upper()}: {t.weight:g}%" for t in params.targets)
    pdf.multi_cell(0, 5, _t(f"Alocação alvo: {pesos}"), **_LINE)
    if meta.get("warnings"):
        pdf.multi_cell(0, 5, _t("Avisos: " + ", ".join(meta["warnings"])), **_LINE)
    pdf.ln(4)


def _initial_balances(pdf: _PlanPDF, data: dict):
    pdf.section("2. Saldos Iniciais Considerados")

    def _draw_header():
        pdf.set_font("helvetica", "B", 8)
        pdf.set_fill_color(230, 237, 245)
        pdf.set_draw_color(180, 180, 180)
        pdf.cell(40, 6, "Ativo alvo", border=1, fill=True)
        pdf.cell(110, 6, _t("Composição do saldo inicial"), border=1, fill=True)
        pdf.cell(40, 6, "Saldo inicial total", border=1, fill=True, align="R", **_LINE)

    if pdf.will_page_break(11) and pdf.get_y() > 27:
        pdf.add_page()
    _draw_header()

    pdf.set_font("helvetica", "", 8)
    for b in data.get("initial_balances", []):
        if pdf.will_page_break(5):
            pdf.add_page()
            _draw_header()
            pdf.set_font("helvetica", "", 8)
        parts = [f"Carteira ({_brl(b['direct'])})"] + [f"{c['ativo']} ({_brl(c['saldo'])})" for c in b.get("counted", [])]
        pdf.cell(40, 5, _t(b["ticker"]), border=1)
        pdf.cell(110, 5, _t(" + ".join(parts))[:95], border=1)
        pdf.cell(40, 5, _t(_brl(b["total"])), border=1, align="R", **_LINE)
    pdf.ln(5)


def _month_height(m: dict) -> float:
    """Altura em mm do bloco do mês: título + cabeçalhos + linhas + IR estimado + espaçamento."""
    sales = m.get("sales") or m.get("vendas") or []
    purchases = m.get("purchases") or m.get("compras") or []
    n_rows = max(len(sales), len(purchases), 1)
    # Título do mês: 6mm
    # Cabeçalho das tabelas (VENDER/COMPRAR): 5mm título + 4mm colunas = 9mm
    # Linhas de dados: n_rows * 4mm
    # Pós-tabela gap: 1mm
    # IR estimado: 4mm
    # Espaçamento final: 3mm (pdf.ln(3))
    h = 6.0 + 9.0 + n_rows * 4.0 + 1.0 + 4.0 + 3.0
    warnings = m.get("warnings") or []
    if warnings:
        h += len(warnings) * 4.0
    return float(h)


def _month(pdf: _PlanPDF, m: dict, data: dict):
    if pdf.will_page_break(_month_height(m)) and pdf.get_y() > 27:
        pdf.add_page()
    sold = m["exempt_sales_total"] + m["taxed_sales_total"]
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 6, _t(f"MÊS {m['month']} - Total a vender: {_brl(sold)} | Total a comprar: {_brl(m['purchase_total'])}"),
             **_LINE)
    pdf.set_text_color(0, 0, 0)
    y_start = pdf.get_y()

    pdf.set_font("helvetica", "B", 7)
    pdf.set_fill_color(254, 235, 235)
    pdf.cell(92, 5, "VENDER", border=1, fill=True, **_LINE)
    pdf.cell(30, 4, "Ativo", border=1)
    pdf.cell(16, 4, "Qtd", border=1, align="C")
    pdf.cell(24, 4, "Valor", border=1, align="R")
    pdf.cell(22, 4, "IR estimado", border=1, align="R", **_LINE)
    pdf.set_font("helvetica", "", 7)
    if not m["sales"]:
        pdf.cell(92, 4, "Nenhuma venda neste mês", border=1, **_LINE)
    for s in m["sales"]:
        pdf.cell(30, 4, _t(s["ativo"]), border=1)
        pdf.cell(16, 4, _qtd(s["qtd"]), border=1, align="C")
        pdf.cell(24, 4, _t(_brl(s["valor"])), border=1, align="R")
        pdf.cell(22, 4, _t(_brl(s["ir_estimado"])), border=1, align="R", **_LINE)
    y_end_sales = pdf.get_y()

    pdf.set_y(y_start)
    pdf.set_x(108)
    pdf.set_font("helvetica", "B", 7)
    pdf.set_fill_color(240, 253, 244)
    pdf.cell(92, 5, "COMPRAR (rebalanceado)", border=1, fill=True, new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_x(108)
    for w, label, align in ((18, "Ativo", "L"), (16, "Cotac.", "R"), (12, "Qtd", "C"),
                            (18, "Compra", "R"), (18, "Saldo fin.", "R"), (10, "Part.", "R")):
        pdf.cell(w, 4, _t(label), border=1, align=align)
    pdf.ln()
    pdf.set_font("helvetica", "", 7)
    for p in m["purchases"]:
        pdf.set_x(108)
        pdf.cell(18, 4, _t(p["ticker"]), border=1)
        pdf.cell(16, 4, _t(_brl(p["price"])), border=1, align="R")
        pdf.cell(12, 4, str(p["qtd_sugerida"]) if p["qtd_sugerida"] else "-", border=1, align="C")
        pdf.cell(18, 4, _t(_brl(p["valor"])) if p["valor"] > 0 else "-", border=1, align="R")
        pdf.cell(18, 4, _t(_brl(p["saldo_final"])), border=1, align="R")
        pdf.cell(10, 4, f"{p['participacao_final'] * 100:.1f}%", border=1, align="R")
        pdf.ln()
    y_end_buys = pdf.get_y()

    pdf.set_y(max(y_end_sales, y_end_buys) + 1)
    pdf.set_font("helvetica", "B", 7)
    pdf.cell(0, 4, _t(f"IR estimado do mês: {_brl(m['ir_estimado_total'])}"), **_LINE)
    if m["warnings"]:
        pdf.set_font("helvetica", "I", 7)
        pdf.set_text_color(160, 60, 30)
        pdf.multi_cell(0, 4, _t("Avisos: " + ", ".join(m["warnings"])), **_LINE)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def render_plan_pdf(
    envelope: dict, params: PlanRequest, *, generated_at: dt.datetime, compress: bool = True
) -> bytes:
    data, meta = envelope["data"], envelope["meta"]
    pdf = _PlanPDF(generated_at)
    pdf.compress = compress
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    _parameters(pdf, params, meta)
    _initial_balances(pdf, data)
    n = data["totals"]["months"]
    title = "3. Cronograma de Aportes" if params.contributions_only else (
        f"3. Cronograma de Migração ({n} {'mês' if n == 1 else 'meses'})"
    )
    pdf.section(title)
    for m in data["months"]:
        _month(pdf, m, data)

    totals = data["totals"]
    if pdf.will_page_break(15) and pdf.get_y() > 27:
        pdf.add_page()
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(0, 6, _t(f"IR estimado total: {_brl(totals['ir_estimado'])}"), **_LINE)
    if totals["ir_not_estimated"]:
        pdf.set_font("helvetica", "", 8)
        pdf.multi_cell(0, 5, _t("IR não estimado (regras próprias): " + ", ".join(totals["ir_not_estimated"])),
                       **_LINE)
    return bytes(pdf.output())
