import pytest
from django.contrib.auth import get_user_model

from apps.accounts.models import User

pytestmark = pytest.mark.django_db


def test_get_user_model_is_custom():
    assert get_user_model() is User


def test_superuser_flags():
    su = User.objects.create_superuser(
        username="root", email="r@example.com", password="pw-abc-12345"
    )
    assert su.is_staff and su.is_superuser
