from django.http import HttpRequest
from ninja import Router, Status

from apps.api.errors import ApiError
from apps.api.schemas import TokenIn, TokenStatusOut
from apps.core.models import UserToken
from apps.market.services.sources import dlp

router = Router(tags=["dlp"])


def _status(ut: UserToken | None) -> dict:
    return {"configured": ut is not None, "updated_at": ut.updated_at if ut else None}


@router.post("/token", response=TokenStatusOut)
def save_token(request: HttpRequest, payload: TokenIn):
    token = payload.token.strip()
    if not token:
        raise ApiError(422, "invalid_token", "Informe o token DLP.")
    ut = UserToken.objects.filter(user=request.user).first() or UserToken(user=request.user)
    old = ut.get_token() if ut.pk else None
    ut.set_token(token)
    ut.save()
    if old and old != token:
        dlp.invalidate_user_cache(old)
    return _status(ut)


@router.get("/token-status", response=TokenStatusOut)
def token_status(request: HttpRequest):
    return _status(UserToken.objects.filter(user=request.user).first())


@router.delete("/token", response={204: None})
def delete_token(request: HttpRequest):
    ut = UserToken.objects.filter(user=request.user).first()
    if ut is not None:
        old = ut.get_token()
        ut.delete()
        dlp.invalidate_user_cache(old)
    return Status(204, None)
