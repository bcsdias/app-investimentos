import logging

from cryptography.fernet import InvalidToken
from django.db import IntegrityError, transaction
from django.http import HttpRequest
from ninja import Router, Status

from apps.api.errors import ApiError
from apps.api.schemas import TokenIn, TokenStatusOut
from apps.core import security
from apps.core.models import UserToken
from apps.market.services.sources import dlp

router = Router(tags=["dlp"])
logger = logging.getLogger(__name__)


def _old_token(ut: UserToken) -> str | None:
    """Token antigo em claro, ou None se ilegível (não bloqueia troca/remoção)."""
    try:
        return ut.get_token()
    except InvalidToken:
        logger.warning("Token DLP ilegível para o usuário %s; será sobrescrito.", ut.user_id)
        return None


def _status(ut: UserToken | None) -> dict:
    return {"configured": ut is not None, "updated_at": ut.updated_at if ut else None}


@router.post("/token", response=TokenStatusOut)
def save_token(request: HttpRequest, payload: TokenIn):
    token = payload.token.strip()
    if not token:
        raise ApiError(422, "invalid_token", "Informe o token DLP.")
    existing = UserToken.objects.filter(user=request.user).first()
    old = _old_token(existing) if existing else None
    encrypted = security.encrypt(token)
    try:
        with transaction.atomic():
            ut, _ = UserToken.objects.update_or_create(
                user=request.user,
                defaults={"encrypted_token": encrypted},
            )
    except IntegrityError:
        with transaction.atomic():
            ut = UserToken.objects.get(user=request.user)
            if old is None:
                old = _old_token(ut)
            ut.encrypted_token = encrypted
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
        old = _old_token(ut)
        ut.delete()
        if old:
            dlp.invalidate_user_cache(old)
    return Status(204, None)
