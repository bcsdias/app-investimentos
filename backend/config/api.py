from ninja import NinjaAPI
from ninja.security import django_auth

from apps.accounts.api import router as accounts_router
from apps.api.errors import ApiError

api = NinjaAPI(
    title="app-investimentos API",
    version="1",
    auth=django_auth,
    urls_namespace="api-v1",
    docs_url="/docs",
)


@api.exception_handler(ApiError)
def handle_api_error(request, exc: ApiError):
    return api.create_response(
        request,
        {"code": exc.code, "message": exc.message, "details": exc.details},
        status=exc.status,
    )

api.add_router("/auth/", accounts_router)
