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
