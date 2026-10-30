import pytest
from django.core.cache import cache
from django.test import RequestFactory

from apps.accounts import ratelimit

URL = "/api/v1/auth/login"


def test_key_for_normalization():
    k1 = ratelimit.key_for(" Bruno ", "127.0.0.1")
    k2 = ratelimit.key_for("bruno", "127.0.0.1")
    k3 = ratelimit.key_for("BRUNO", "127.0.0.1")
    assert k1 == k2 == k3
    assert "bruno" in k1
    assert "127.0.0.1" in k1


def test_client_ip_direct_and_proxies(settings):
    rf = RequestFactory()

    # Direct client (not in trusted proxies)
    req1 = rf.get("/", REMOTE_ADDR="198.51.100.1")
    assert ratelimit.client_ip(req1) == "198.51.100.1"

    # Spoofed X-Real-IP from untrusted client is ignored
    req2 = rf.get("/", REMOTE_ADDR="198.51.100.1", HTTP_X_REAL_IP="203.0.113.5")
    assert ratelimit.client_ip(req2) == "198.51.100.1"

    # Trusted proxy 127.0.0.1 forwards X-Real-IP
    req3 = rf.get("/", REMOTE_ADDR="127.0.0.1", HTTP_X_REAL_IP="203.0.113.5")
    assert ratelimit.client_ip(req3) == "203.0.113.5"

    # Trusted proxy ::1 forwards X-Real-IP
    req4 = rf.get("/", REMOTE_ADDR="::1", HTTP_X_REAL_IP="203.0.113.5")
    assert ratelimit.client_ip(req4) == "203.0.113.5"

    # Private network / Docker bridge proxy (172.16.0.0/12) forwards X-Real-IP by default
    req_docker = rf.get("/", REMOTE_ADDR="172.18.0.3", HTTP_X_REAL_IP="203.0.113.7")
    assert ratelimit.client_ip(req_docker) == "203.0.113.7"

    # Custom trusted proxy in settings
    settings.TRUSTED_PROXIES = {"10.0.0.1"}
    req5 = rf.get("/", REMOTE_ADDR="10.0.0.1", HTTP_X_REAL_IP="203.0.113.99")
    assert ratelimit.client_ip(req5) == "203.0.113.99"

    # When TRUSTED_PROXIES is explicitly set, other IPs are not trusted
    req_untrusted_docker = rf.get("/", REMOTE_ADDR="172.18.0.3", HTTP_X_REAL_IP="203.0.113.7")
    assert ratelimit.client_ip(req_untrusted_docker) == "172.18.0.3"


def test_ratelimit_unit_methods(locmem_cache):
    key = ratelimit.key_for("alice", "10.0.0.1")
    assert not ratelimit.is_blocked(key)

    for _ in range(ratelimit.MAX_FAILURES - 1):
        ratelimit.register_failure(key)
        assert not ratelimit.is_blocked(key)

    ratelimit.register_failure(key)
    assert ratelimit.is_blocked(key)

    ratelimit.reset(key)
    assert not ratelimit.is_blocked(key)


def test_sixth_attempt_is_blocked(client, user, locmem_cache):
    # 5 falhas consecutivas com senha errada
    for _ in range(5):
        r = client.post(URL, {"username": "bruno", "password": "wrong"}, content_type="application/json")
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_credentials"

    # 6ª tentativa, mesmo com senha correta, é bloqueada com 429
    r6 = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r6.status_code == 429
    data = r6.json()
    assert data["code"] == "too_many_attempts"
    assert "Muitas tentativas de login" in data["message"]


def test_other_user_or_ip_not_blocked(client, user, other_user, locmem_cache):
    ip1 = "192.168.1.10"
    ip2 = "192.168.1.20"

    # 5 falhas de bruno no ip1
    for _ in range(5):
        r = client.post(URL, {"username": "bruno", "password": "wrong"}, content_type="application/json", REMOTE_ADDR=ip1)
        assert r.status_code == 401

    # bruno no ip1 está bloqueado
    r_blocked = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json", REMOTE_ADDR=ip1)
    assert r_blocked.status_code == 429

    # outro usuário (ana) no mesmo ip1 NÃO está bloqueado
    r_ana = client.post(URL, {"username": "ana", "password": "s3nha-forte!"}, content_type="application/json", REMOTE_ADDR=ip1)
    assert r_ana.status_code == 200

    # mesmo usuário (bruno) em outro ip2 NÃO está bloqueado
    r_bruno_ip2 = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json", REMOTE_ADDR=ip2)
    assert r_bruno_ip2.status_code == 200


def test_success_resets_counter(client, user, locmem_cache):
    # 4 falhas
    for _ in range(4):
        r = client.post(URL, {"username": "bruno", "password": "wrong"}, content_type="application/json")
        assert r.status_code == 401

    # 1 sucesso
    r_ok = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r_ok.status_code == 200

    # mais 4 falhas -> total acumulado após reset é 4, ainda não atinge 5
    for _ in range(4):
        r = client.post(URL, {"username": "bruno", "password": "wrong"}, content_type="application/json")
        assert r.status_code == 401

    # próxima tentativa com senha certa ainda funciona
    r_ok2 = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r_ok2.status_code == 200


def test_cache_down_does_not_block(client, user, monkeypatch):
    def broken_cache(*args, **kwargs):
        raise RuntimeError("Redis connection lost")

    monkeypatch.setattr(cache, "get", broken_cache)
    monkeypatch.setattr(cache, "add", broken_cache)
    monkeypatch.setattr(cache, "incr", broken_cache)
    monkeypatch.setattr(cache, "delete", broken_cache)

    # Com cache quebrado, login com credenciais corretas deve funcionar normalmente
    r = client.post(URL, {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r.status_code == 200
    assert r.json()["username"] == "bruno"

    # E login incorreto retorna 401 sem estourar 500
    r_bad = client.post(URL, {"username": "bruno", "password": "wrong"}, content_type="application/json")
    assert r_bad.status_code == 401
    assert r_bad.json()["code"] == "invalid_credentials"
