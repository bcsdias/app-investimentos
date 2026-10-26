"""Normaliza o payload /resumo da DLP em uma lista de Holding (puro, sem Django).

Sempre usa price_fx / pm_fx (BRL). Mesmo `ativo` em várias linhas (corretoras) é agregado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Holding:
    ativo: str
    classe: str
    qtd: float
    price: float | None  # price_fx (BRL); None = sem cotação
    pm: float | None  # pm_fx (BRL)
    vlr_investido: float
    market_cod: str
    moeda: str

    @property
    def saldo(self) -> float:
        return self.qtd * self.price if self.price else self.vlr_investido

    @property
    def saldo_source(self) -> Literal["market", "cost"]:
        return "market" if self.price else "cost"


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN -> None


def _positive(v) -> float | None:
    n = _num(v)
    return n if n is not None and n > 0 else None


def _aggregate(ativo: str, rows: list[dict]) -> tuple[Holding, bool]:
    qtds = [_num(r.get("qtd", r.get("quantidade"))) or 0.0 for r in rows]
    qtd = sum(qtds)
    vlr = sum(_num(r.get("vlr_investido")) or 0.0 for r in rows)
    price = next((p for p in (_positive(r.get("price_fx")) for r in rows) if p is not None), None)

    pm_rows = [(q, _positive(r.get("pm_fx"))) for q, r in zip(qtds, rows)]
    pm_rows = [(q, pm) for q, pm in pm_rows if pm is not None and q > 0]
    pm_qtd = sum(q for q, _ in pm_rows)
    pm = sum(q * pm for q, pm in pm_rows) / pm_qtd if pm_qtd else None

    main = rows[max(range(len(rows)), key=lambda i: qtds[i])]
    classes = {str(r.get("classe") or "").strip() for r in rows}
    holding = Holding(
        ativo=ativo,
        classe=str(main.get("classe") or "").strip(),
        qtd=qtd,
        price=price,
        pm=pm,
        vlr_investido=vlr,
        market_cod=str(main.get("market_cod") or "").strip(),
        moeda=str(main.get("moeda") or "").strip(),
    )
    return holding, len(classes) > 1


def normalize_wallet(payload: dict) -> tuple[list[Holding], list[str]]:
    """Holdings com saldo > 0, por saldo decrescente; warnings globais (mixed_class:<ativo>)."""
    grouped: dict[str, list[dict]] = {}
    for row in (payload or {}).get("wallet") or []:
        ativo = str(row.get("ativo") or "").strip()
        if ativo:
            grouped.setdefault(ativo, []).append(row)

    holdings: list[Holding] = []
    warnings: list[str] = []
    for ativo, rows in grouped.items():
        h, mixed = _aggregate(ativo, rows)
        if mixed:
            warnings.append(f"mixed_class:{ativo}")
        if h.saldo > 0:
            holdings.append(h)
    holdings.sort(key=lambda h: (-h.saldo, h.ativo))
    return holdings, warnings
