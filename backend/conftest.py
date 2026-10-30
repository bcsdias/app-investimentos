import pytest
from django.core.cache import cache
from django.test import Client


@pytest.fixture
def locmem_cache(settings):
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "api-tests",
        }
    }
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture
def user(db):
    from django.contrib.auth import get_user_model

    return get_user_model().objects.create_user(username="bruno", password="s3nha-forte!")


@pytest.fixture
def other_user(db):
    from django.contrib.auth import get_user_model

    return get_user_model().objects.create_user(username="ana", password="s3nha-forte!")


@pytest.fixture
def auth_client(user):
    client = Client()
    client.force_login(user)
    return client


@pytest.fixture
def user_with_token(user):
    from apps.core.models import UserToken

    ut = UserToken(user=user)
    ut.set_token("tok-teste")
    ut.save()
    return user


@pytest.fixture
def user_with_token_client(user_with_token):
    client = Client()
    client.force_login(user_with_token)
    return client


@pytest.fixture
def assets_csv(settings):
    from pathlib import Path

    from apps.market import assets

    settings.MARKET_ASSETS_CSV = Path(__file__).resolve().parent / "apps/market/tests/fixtures/ativos_mini.csv"
    assets.load_universe.cache_clear()
    yield
    assets.load_universe.cache_clear()
