"""Ambiente do harness: sys.path, Django, matplotlib Agg e isolamento do legado.

Importa src: sai junto com o legado na Task 9 da Fase 6.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

_LEGACY_LOGGER = "invest_app"  # nome usado por src/utils/logger.py::setup_logger
_UPSTASH_VARS = ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")


def configure() -> None:
    for p in (BACKEND_DIR, REPO_ROOT, REPO_ROOT / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import matplotlib

    matplotlib.use("Agg")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()
    isolate_legacy()


def isolate_legacy() -> None:
    """Legado só com cache em memória e log no console (nada em log/ nem no Upstash)."""
    import matplotlib

    matplotlib.use("Agg")

    # setup_logger() retorna cedo se o logger já tem handlers: assim o
    # RotatingFileHandler (log/web_app.log) nem chega a ser criado.
    legacy_logger = logging.getLogger(_LEGACY_LOGGER)
    if not legacy_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("legado %(levelname)s: %(message)s"))
        legacy_logger.addHandler(handler)
        legacy_logger.setLevel(logging.WARNING)
        legacy_logger.propagate = False

    import src.data.cache as legacy_cache  # roda load_dotenv() do legado
    from src.utils.logger import logger

    for var in _UPSTASH_VARS:
        os.environ.pop(var, None)
    legacy_cache._get_redis_client = lambda: None
    for h in list(logger.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            logger.removeHandler(h)
            h.close()
