from django.conf import settings
from django.db import models

from apps.core import security


class UserToken(models.Model):
    """Token da API DLP de um usuário, cifrado em repouso com Fernet."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dlp_token",
    )
    encrypted_token = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_token(self, raw: str) -> None:
        self.encrypted_token = security.encrypt(raw)

    def get_token(self) -> str:
        return security.decrypt(self.encrypted_token)

    def __str__(self) -> str:
        return f"UserToken<{self.user}>"


class MarketSeries(models.Model):
    """Um ponto diário de uma série de mercado (benchmark, índice, taxa)."""

    series_key = models.CharField(max_length=120, db_index=True)  # ex.: "bcb:12", "yf:^BVSP"
    source = models.CharField(max_length=16)  # BCB | YF | TD | B3 | PTAX
    reference_date = models.DateField()
    value = models.DecimalField(max_digits=20, decimal_places=8)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["series_key", "reference_date"], name="uniq_series_point"
            )
        ]
        indexes = [models.Index(fields=["series_key", "reference_date"])]
        ordering = ["series_key", "reference_date"]

    def __str__(self) -> str:
        return f"{self.series_key}@{self.reference_date}={self.value}"
