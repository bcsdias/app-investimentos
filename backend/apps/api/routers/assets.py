from __future__ import annotations

from django.http import HttpRequest
from ninja import Router

from apps.api.errors import ApiError
from apps.api.schemas import AssetSearchOut
from apps.market import assets

router = Router(tags=["assets"])


@router.get("/search", response=list[AssetSearchOut])
def search_assets(request: HttpRequest, q: str = "", limit: int = 20):
    term = q.strip()
    if len(term) < 2:
        raise ApiError(
            422,
            "invalid_query",
            "Informe pelo menos 2 caracteres para a busca.",
        )
    if not (1 <= limit <= 50):
        raise ApiError(
            422,
            "invalid_query",
            "O parâmetro limit deve estar entre 1 e 50.",
        )
    return [
        {
            "sigla": a.sigla,
            "nome": a.nome,
            "classe": a.classe,
            "mercado": a.mercado,
            "yf_ticker": a.yf_ticker,
            "brl_variant": a.brl_variant,
        }
        for a in assets.search(term, limit=limit)
    ]
