import datetime as dt

import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from apps.core.models import MarketSeries, UserToken

pytestmark = pytest.mark.django_db


def _user(username="alice"):
    return get_user_model().objects.create_user(username=username, password="pw-abc-12345")


def test_usertoken_roundtrip_persisted():
    tok = UserToken(user=_user())
    tok.set_token("DLP-secret-xyz")
    tok.save()
    assert UserToken.objects.get(pk=tok.pk).get_token() == "DLP-secret-xyz"


def test_usertoken_column_is_not_plaintext():
    tok = UserToken(user=_user())
    tok.set_token("DLP-secret-xyz")
    tok.save()
    assert "DLP-secret-xyz" not in UserToken.objects.get(pk=tok.pk).encrypted_token


def test_usertoken_is_one_per_user():
    user = _user()
    UserToken.objects.create(user=user, encrypted_token="x")
    with pytest.raises(IntegrityError):
        UserToken.objects.create(user=user, encrypted_token="y")


def test_usertoken_related_name():
    tok = UserToken(user=_user())
    tok.set_token("DLP-1")
    tok.save()
    assert tok.user.dlp_token.get_token() == "DLP-1"


def test_marketseries_unique_point():
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 2), value="0.00043739",
    )
    with pytest.raises(IntegrityError):
        MarketSeries.objects.create(
            series_key="bcb:12", source="BCB",
            reference_date=dt.date(2024, 1, 2), value="0.00099999",
        )


def test_marketseries_allows_same_key_other_date():
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 2), value="0.0004",
    )
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 3), value="0.0004",
    )
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2
