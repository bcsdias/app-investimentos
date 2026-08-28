"""Criptografia simétrica (Fernet) para os tokens DLP em repouso.

Porta de src/data/user_store.py::_get_cipher — MESMA chave, para que os tokens
cifrados pelo app Streamlit legado continuem decifráveis após a migração.
"""
from cryptography.fernet import Fernet
from django.conf import settings


def _cipher() -> Fernet:
    key = settings.FERNET_KEY
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _cipher().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _cipher().decrypt(ciphertext.encode()).decode()
