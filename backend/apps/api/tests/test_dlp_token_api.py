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
