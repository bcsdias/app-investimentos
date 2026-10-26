from django.contrib.auth import authenticate, login, logout
from django.http import HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from ninja import Router, Schema, Status
from ninja.decorators import decorate_view

from apps.api.errors import ApiError
from apps.core.models import UserToken

router = Router(tags=["auth"])


class LoginIn(Schema):
    username: str
    password: str


class UserOut(Schema):
    id: int
    username: str
    email: str


class MeOut(UserOut):
    has_dlp_token: bool


@router.get("/csrf", auth=None, response={204: None})
@decorate_view(ensure_csrf_cookie)
def csrf(request: HttpRequest):
    return Status(204, None)


@router.post("/login", auth=None, response=UserOut)
@decorate_view(csrf_protect)
def login_view(request: HttpRequest, payload: LoginIn):
    user = authenticate(request, username=payload.username, password=payload.password)
    if user is None:
        raise ApiError(401, "invalid_credentials", "Usuário ou senha inválidos.")
    login(request, user)
    return user


@router.post("/logout", response={204: None})
def logout_view(request: HttpRequest):
    logout(request)
    return Status(204, None)


@router.get("/me", response=MeOut)
def me(request: HttpRequest):
    user = request.user
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "has_dlp_token": UserToken.objects.filter(user=user).exists(),
    }
