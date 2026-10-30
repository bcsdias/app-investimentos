# Generated manually on 2026-10-30 for Fase 7c.1

from django.db import migrations

REMOVED_KEYS = (
    "bcb:12461",
    "bcb:12466",
    "bcb:12467",
    "bcb:12468",
    "bcb:12469",
)


def forwards(apps, schema_editor):
    MarketSeries = apps.get_model("core", "MarketSeries")
    MarketSeries.objects.filter(series_key__in=REMOVED_KEYS).delete()
    from apps.market.cache import bump_series_generation

    bump_series_generation()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0002_marketseries_kind"),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
