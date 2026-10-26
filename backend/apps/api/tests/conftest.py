import pytest


@pytest.fixture(autouse=True)
def _api_locmem_cache(locmem_cache):
    """Testes da API nunca tocam o Redis real."""
    return locmem_cache
