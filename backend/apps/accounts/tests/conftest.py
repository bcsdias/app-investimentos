import pytest


@pytest.fixture(autouse=True)
def _api_locmem_cache(locmem_cache):
    """Nunca toca o Redis real nos testes da API de contas."""
    yield
