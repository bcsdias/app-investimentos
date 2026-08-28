from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("api/v1/", api.urls)  # Django Ninja — adicionado na Fase 3
]
