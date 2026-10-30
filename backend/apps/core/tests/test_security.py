import base64

import pytest
from cryptography.fernet import Fernet, InvalidToken

from apps.core import security


def test_encrypt_decrypt_roundtrip():
    assert security.decrypt(security.encrypt("DLP-abc123")) == "DLP-abc123"


def test_ciphertext_differs_from_plaintext():
    plain = "DLP-abc123"
    assert security.encrypt(plain) != plain


def test_decrypt_rejects_garbage():
    with pytest.raises(InvalidToken):
        security.decrypt("isto-nao-e-um-token-fernet")


def test_decrypt_rejects_token_from_other_key():
    foreign = Fernet(Fernet.generate_key()).encrypt(b"DLP-abc123").decode()
    with pytest.raises(InvalidToken):
        security.decrypt(foreign)


def test_configured_key_is_32_bytes_base64():
    raw = security._cipher()  # noqa: SLF001 — sanity check do wrapper
    assert isinstance(raw, Fernet)
    from django.conf import settings

    key = settings.FERNET_KEY
    key = key.encode() if isinstance(key, str) else key
    assert len(base64.urlsafe_b64decode(key)) == 32
