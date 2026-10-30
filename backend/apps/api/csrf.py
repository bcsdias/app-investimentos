from django.http import JsonResponse
from django.views.csrf import csrf_failure as django_csrf_failure
from ninja.security import SessionAuth
from ninja.utils import check_csrf

from apps.api.errors import ApiError

CSRF_MESSAGE = "Falha na verificação CSRF. Recarregue a página."
API_PREFIX = "/api/"


def csrf_failed() -> ApiError:
    return ApiError(403, "csrf_failed", CSRF_MESSAGE)


class ApiSessionAuth(SessionAuth):
    """SessionAuth do Ninja, mas a falha de CSRF usa o envelope padrão de erro."""

    def _get_key(self, request):
        if not getattr(request, "_ninja_csrf_exempt", False) and check_csrf(request):
            raise csrf_failed()
        return request.COOKIES.get(self.param_name)


def csrf_failure_view(request, reason="", **kwargs):
    """CSRF_FAILURE_VIEW: JSON padrão sob /api/, página do Django nas demais rotas (admin)."""
    if request.path.startswith(API_PREFIX):
        exc = csrf_failed()
        return JsonResponse(
            {"code": exc.code, "message": exc.message, "details": exc.details}, status=403
        )
    return django_csrf_failure(request, reason=reason, **kwargs)
