import pytest
from django.core.cache import cache
from django.db import connection

pytestmark = pytest.mark.django_db


def test_database_is_appinvest():
    with connection.cursor() as cur:
        cur.execute("SELECT current_database()")
        assert "appinvest" in cur.fetchone()[0]


def test_cache_set_get():
    try:
        cache.set("smoke-key", 123, timeout=10)
    except Exception as exc:  # Redis fora do ar
        pytest.skip(f"Redis indisponível: {exc}")
    assert cache.get("smoke-key") == 123
