from django.test import Client

from apps.core.models import UserToken

URL = "/api/v1/auth"
CSRF_BODY = {
    "code": "csrf_failed",
    "message": "Falha na verificação CSRF. Recarregue a página.",
    "details": None,
}


def test_csrf_sets_cookie(client):
    r = client.get(f"{URL}/csrf")
    assert r.status_code == 204 and "csrftoken" in r.cookies


def test_login_ok_then_me(client, user):
    r = client.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r.status_code == 200 and r.json()["username"] == "bruno"
    assert client.get(f"{URL}/me").json() == {"id": user.id, "username": "bruno", "email": user.email, "has_dlp_token": False}


def test_login_invalid(client, user):
    r = client.post(f"{URL}/login", {"username": "bruno", "password": "errada"}, content_type="application/json")
    assert r.status_code == 401 and r.json()["code"] == "invalid_credentials"


def test_login_requires_csrf(user):
    c = Client(enforce_csrf_checks=True)
    r = c.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r.status_code == 403
    assert r.json() == CSRF_BODY


def test_login_with_csrf_header_ok(user):
    c = Client(enforce_csrf_checks=True)
    token = c.get(f"{URL}/csrf").cookies["csrftoken"].value
    r = c.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"},
               content_type="application/json", HTTP_X_CSRFTOKEN=token)
    assert r.status_code == 200


def test_me_unauthenticated_401(client, db):
    assert client.get(f"{URL}/me").status_code == 401


def test_me_has_dlp_token_true(auth_client, user):
    ut = UserToken(user=user); ut.set_token("tok"); ut.save()
    assert auth_client.get(f"{URL}/me").json()["has_dlp_token"] is True


def test_logout_ends_session(auth_client):
    assert auth_client.post(f"{URL}/logout").status_code == 204
    assert auth_client.get(f"{URL}/me").status_code == 401


def test_logout_without_csrf_is_403(user):
    c = Client(enforce_csrf_checks=True); c.force_login(user)
    r = c.post(f"{URL}/logout")
    assert r.status_code == 403
    assert r.json() == CSRF_BODY


def test_admin_csrf_failure_keeps_django_page(db):
    r = Client(enforce_csrf_checks=True).post("/admin/login/", {"username": "x", "password": "y"})
    assert r.status_code == 403 and r["Content-Type"].startswith("text/html")
