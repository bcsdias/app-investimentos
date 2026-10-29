from django.test import Client

from apps.core.models import UserToken

URL = "/api/v1/dlp"


def _post(c, tok):
    return c.post(f"{URL}/token", {"token": tok}, content_type="application/json")


def test_save_token_encrypts_and_reports_status(auth_client, user):
    r = _post(auth_client, "  abc123\n")
    assert r.status_code == 200 and r.json()["configured"] is True
    ut = UserToken.objects.get(user=user)
    assert ut.get_token() == "abc123" and "abc123" not in ut.encrypted_token
    assert "abc123" not in r.content.decode()


def test_blank_token_is_422(auth_client):
    r = _post(auth_client, "   ")
    assert r.status_code == 422 and r.json()["code"] == "invalid_token"


def test_replace_invalidates_old_token_cache(auth_client, monkeypatch):
    seen = []
    monkeypatch.setattr("apps.api.routers.dlp.dlp.invalidate_user_cache", seen.append)
    _post(auth_client, "old")
    _post(auth_client, "new")
    assert seen == ["old"]


def test_status_without_token(auth_client):
    assert auth_client.get(f"{URL}/token-status").json() == {"configured": False, "updated_at": None}


def test_delete_is_idempotent(auth_client):
    _post(auth_client, "x")
    assert auth_client.delete(f"{URL}/token").status_code == 204
    assert auth_client.delete(f"{URL}/token").status_code == 204
    assert auth_client.get(f"{URL}/token-status").json()["configured"] is False


def test_isolation_between_users(auth_client, other_user):
    _post(auth_client, "x")
    c = Client()
    c.force_login(other_user)
    assert c.get(f"{URL}/token-status").json()["configured"] is False


def test_unauthenticated_401(client, db):
    assert client.get(f"{URL}/token-status").status_code == 401


def test_save_without_csrf_is_403(user):
    c = Client(enforce_csrf_checks=True)
    c.force_login(user)
    assert _post(c, "x").status_code == 403


def test_post_replaces_unreadable_stored_token(auth_client, user):
    _post(auth_client, "old")
    UserToken.objects.filter(user=user).update(encrypted_token="lixo")
    r = _post(auth_client, "new")
    assert r.status_code == 200
    assert UserToken.objects.get(user=user).get_token() == "new"


def test_delete_removes_unreadable_stored_token(auth_client, user):
    _post(auth_client, "old")
    UserToken.objects.filter(user=user).update(encrypted_token="lixo")
    assert auth_client.delete(f"{URL}/token").status_code == 204
    assert not UserToken.objects.filter(user=user).exists()


def test_save_token_survives_concurrent_insert(auth_client, user, monkeypatch):
    from apps.core.models import UserToken

    _post(auth_client, "old_token")

    # Simula a corrida: o router acha que não existe token (.first() -> None)
    # mas o registro já existe no banco
    monkeypatch.setattr(
        UserToken.objects,
        "first",
        lambda: None,
    )
    # No router atual: ut = UserToken.objects.filter(...).first() or UserToken(user=...)
    # Vai instanciar novo UserToken e dar IntegrityError no save().
    orig_filter = UserToken.objects.filter

    class FakeQS:
        def __init__(self, qs):
            self.qs = qs
        def first(self):
            return None
        def __getattr__(self, name):
            return getattr(self.qs, name)

    monkeypatch.setattr(
        UserToken.objects,
        "filter",
        lambda **kw: FakeQS(orig_filter(**kw)),
    )
    r = _post(auth_client, "concurrent_winner")
    assert r.status_code == 200
    assert r.json()["configured"] is True
    assert UserToken.objects.filter(user=user).count() == 1
    assert UserToken.objects.get(user=user).get_token() == "concurrent_winner"


def test_save_token_survives_integrity_error_on_update_or_create(auth_client, user, monkeypatch):
    from apps.core.models import UserToken
    from django.db import IntegrityError

    _post(auth_client, "old_token")

    first_call = True

    def fail_once(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            raise IntegrityError("unique constraint")
        return UserToken.objects.get(user=user), False

    monkeypatch.setattr(UserToken.objects, "update_or_create", fail_once)
    r = _post(auth_client, "recovered_token")
    assert r.status_code == 200
    assert r.json()["configured"] is True
    assert UserToken.objects.filter(user=user).count() == 1
    assert UserToken.objects.get(user=user).get_token() == "recovered_token"

