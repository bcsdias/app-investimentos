from datetime import datetime

from ninja import Schema


class TokenIn(Schema):
    token: str


class TokenStatusOut(Schema):
    configured: bool
    updated_at: datetime | None = None
