import os

import pytest
from django.test import Client

from apps.core.models import UserToken
from apps.market.services.sources import dlp

pytestmark = pytest.mark.integration


def test_real_migration_wallet(db, user):
    token = os.environ.get("DLP_TEST_TOKEN")
    if not token:
        pytest.skip("DLP_TEST_TOKEN não definido no ambiente")
    ut = UserToken(user=user)
    ut.set_token(token)
    ut.save()
    c = Client()
    c.force_login(user)
    r = c.get("/api/v1/migration/wallet")
    assert r.status_code == 200 and r.json()["data"]
    raw = dlp.buscar_resumo_carteira(token)["wallet"][0]
    print("Campos /resumo wallet:", sorted(raw))
    print("Classes:", sorted({i["classe"] for i in r.json()["data"]}))
