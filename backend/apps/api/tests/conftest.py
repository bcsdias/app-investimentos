import pytest
from django.test import Client

from apps.core.models import UserToken


@pytest.fixture(autouse=True)
def _api_locmem_cache(locmem_cache):
    """Testes da API nunca tocam o Redis real."""
    return locmem_cache


@pytest.fixture
def user_with_token(user):
    ut = UserToken(user=user)
    ut.set_token("tok-teste")
    ut.save()
    return user


@pytest.fixture
def user_with_token_client(user_with_token):
    client = Client()
    client.force_login(user_with_token)
    return client
