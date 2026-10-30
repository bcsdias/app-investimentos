import ipaddress
import logging
from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

logger = logging.getLogger(__name__)

MAX_FAILURES = 5
WINDOW_SECONDS = 900


def key_for(username: str, ip: str) -> str:
    """Retorna a chave de rate limit para o par (usuário, IP).

    O nome de usuário é normalizado em minúsculas e sem espaços ao redor.
    A senha nunca é utilizada na composição da chave.
    """
    norm_user = (username or "").strip().lower()
    norm_ip = (ip or "").strip()
    return f"ratelimit:login:{norm_user}:{norm_ip}"


def is_blocked(key: str) -> bool:
    """Verifica se o limite de tentativas foi atingido.

    Qualquer falha no cache é logada e tratada como 'não bloqueado'.
    """
    try:
        count = cache.get(key)
        if count is None:
            return False
        return int(count) >= MAX_FAILURES
    except Exception:
        logger.exception("Erro ao verificar rate limit no cache para a chave '%s'", key)
        return False


def register_failure(key: str) -> None:
    """Incrementa o contador de falhas no cache.

    Cria a chave com TTL WINDOW_SECONDS se ainda não existir.
    Qualquer erro no cache é logado e ignorado.
    """
    try:
        cache.add(key, 0, timeout=WINDOW_SECONDS)
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, 1, timeout=WINDOW_SECONDS)
    except Exception:
        logger.exception("Erro ao registrar falha de login no cache para a chave '%s'", key)


def reset(key: str) -> None:
    """Reseta o contador de falhas para a chave informada."""
    try:
        cache.delete(key)
    except Exception:
        logger.exception("Erro ao resetar rate limit no cache para a chave '%s'", key)


# Redes privadas e loopback (RFC 1918 + ::1). Prefixo de string não serve: "172." pegaria
# faixas públicas como 172.217.0.0/16; só 172.16.0.0/12 é privada.
_DEFAULT_PROXY_NETS = tuple(
    ipaddress.ip_network(n)
    for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128")
)


def _is_trusted_proxy(remote_addr: str) -> bool:
    if not remote_addr:
        return False
    custom = getattr(settings, "TRUSTED_PROXIES", None)
    if custom is not None:
        return remote_addr in custom
    try:
        addr = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    return any(addr in net for net in _DEFAULT_PROXY_NETS)


def client_ip(request: HttpRequest) -> str:
    """Obtém o IP real do cliente.

    Lê REMOTE_ADDR; se REMOTE_ADDR for de um proxy confiável (definido em
    settings.TRUSTED_PROXIES ou, por padrão, loopback e redes privadas como Docker bridge)
    e HTTP_X_REAL_IP estiver presente, utiliza HTTP_X_REAL_IP.
    """
    meta = getattr(request, "META", {}) or {}
    remote_addr = meta.get("REMOTE_ADDR", "")
    if _is_trusted_proxy(remote_addr) and "HTTP_X_REAL_IP" in meta:
        real_ip = meta.get("HTTP_X_REAL_IP")
        if real_ip:
            return str(real_ip).strip()
    return remote_addr
