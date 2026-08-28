from django.contrib import admin

from .models import MarketSeries, UserToken


@admin.register(UserToken)
class UserTokenAdmin(admin.ModelAdmin):
    list_display = ("user", "updated_at")
    readonly_fields = ("encrypted_token", "created_at", "updated_at")
    search_fields = ("user__username", "user__email")

    def has_add_permission(self, request):
        # tokens entram pela API (Fase 3), nunca digitados no Admin (evita
        # gravar valor não-cifrado no campo encrypted_token).
        return False


@admin.register(MarketSeries)
class MarketSeriesAdmin(admin.ModelAdmin):
    list_display = ("series_key", "source", "reference_date", "value")
    list_filter = ("source",)
    date_hierarchy = "reference_date"
    search_fields = ("series_key",)
    ordering = ("series_key", "reference_date")
