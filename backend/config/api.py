from ninja import NinjaAPI

from apps.accounts.api import router as accounts_router
from apps.api.csrf import ApiSessionAuth
from apps.api.errors import ApiError
from apps.api.routers.dlp import router as dlp_router
from apps.api.routers.portfolio import router as portfolio_router

api = NinjaAPI(
    title="app-investimentos API",
    version="1",
    auth=ApiSessionAuth(),
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
api.add_router("/dlp/", dlp_router)
api.add_router("/portfolio/", portfolio_router)

