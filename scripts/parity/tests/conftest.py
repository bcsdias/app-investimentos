import pytest

from parity import bootstrap


@pytest.fixture(scope="session", autouse=True)
def _isolated_legacy():
    bootstrap.isolate_legacy()
