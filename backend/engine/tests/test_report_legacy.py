import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sem display

from engine import report_legacy  # noqa: E402


def _combined():
    idx = pd.bdate_range("2022-01-03", periods=300)
    return pd.DataFrame(
        {
            "Carteira": np.linspace(100, 130, len(idx)),
            "CDI": np.linspace(100, 118, len(idx)),
        },
        index=idx,
    )


def test_plot_twr_evolution_returns_figure():
    fig = report_legacy.plot_twr_evolution(_combined(), title_suffix="teste")
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) >= 1


def test_plot_drawdown_returns_figure():
    assert report_legacy.plot_drawdown(_combined()).__class__.__name__ == "Figure"


def test_plot_risk_return_returns_figure():
    assert report_legacy.plot_risk_return(_combined()).__class__.__name__ == "Figure"


def test_export_all_writes_pngs(tmp_path):
    from engine.dataset import DatasetResult

    res = DatasetResult(df_combined=_combined(), selic_series=None, portfolio_df=None)
    paths = report_legacy.export_all(res, str(tmp_path))
    assert paths
    for p in paths:
        assert p.endswith(".png")
        assert (tmp_path / p.split("/")[-1]).exists()
