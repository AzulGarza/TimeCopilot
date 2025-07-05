import pandas as pd
import pytest
from utilsforecast.data import generate_series

from ..conftest import models


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
@pytest.mark.parametrize("h", [1, 12])
def test_correct_forecast_dates(model, freq, h):
    n_series = 5
    df = generate_series(
        n_series,
        freq=freq,
    )
    df["unique_id"] = df["unique_id"].astype(str)
    df_test = df.groupby("unique_id").tail(h)
    df_train = df.drop(df_test.index)
    fcst_df = model.forecast(
        df_train,
        h=h,
        freq=freq,
    )
    exp_n_cols = 3
    assert fcst_df.shape == (n_series * h, exp_n_cols)
    exp_cols = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        fcst_df[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        df_test[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
@pytest.mark.parametrize("n_windows", [1, 4])
def test_cross_validation(model, freq, n_windows):
    h = 12
    n_series = 5
    df = generate_series(n_series, freq=freq, equal_ends=True)
    df["unique_id"] = df["unique_id"].astype(str)
    cv_df = model.cross_validation(
        df,
        h=h,
        freq=freq,
        n_windows=n_windows,
    )
    exp_n_cols = 5  # unique_id, cutoff, ds, y, model
    assert cv_df.shape == (n_series * h * n_windows, exp_n_cols)
    cutoffs = cv_df["cutoff"].unique()
    assert len(cutoffs) == n_windows
    df_test = df.groupby("unique_id").tail(h * n_windows)
    exp_cols = ["unique_id", "ds", "y"]
    pd.testing.assert_frame_equal(
        cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        df_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
    )
    if n_windows == 1:
        # test same results using predict with less data
        df_test = df.groupby("unique_id").tail(h)
        df_train = df.drop(df_test.index)
        fcst_df = model.forecast(
            df_train,
            h=h,
            freq=freq,
        )
        exp_cols = ["unique_id", "ds"]
        pd.testing.assert_frame_equal(
            cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
            fcst_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        )


@pytest.mark.parametrize("model", models)
def test_passing_both_level_and_quantiles(model):
    df = generate_series(n_series=1, freq="D")
    with pytest.raises(ValueError):
        model.forecast(
            df=df,
            h=1,
            freq="D",
            level=[80, 95],
            quantiles=[0.1, 0.5, 0.9],
        )
    with pytest.raises(ValueError):
        model.cross_validation(
            df=df,
            h=1,
            freq="D",
            level=[80, 95],
            quantiles=[0.1, 0.5, 0.9],
        )


@pytest.mark.parametrize("model", models)
def test_using_quantiles(model):
    qs = [i * 0.1 for i in range(1, 10)]
    df = generate_series(n_series=2, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=3,
        freq="D",
        quantiles=qs,
    )
    exp_qs_cols = [f"{model.alias}-q-{int(100 * q)}" for q in qs]
    assert all(col in fcst_df.columns for col in exp_qs_cols)
    assert not any(("-lo-" in col or "-hi-" in col) for col in fcst_df.columns)
    # test monotonicity of quantiles
    for c1, c2 in zip(exp_qs_cols[:-1], exp_qs_cols[1:], strict=False):
        if model.alias == "ZeroModel":
            # ZeroModel is a constant model, so all quantiles should be the same
            assert fcst_df[c1].eq(fcst_df[c2]).all()
        elif "chronos-t5" in model.alias.lower():
            # sometimes it gives this condition
            assert fcst_df[c1].le(fcst_df[c2]).all()
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()


@pytest.mark.parametrize("model", models)
def test_using_level(model):
    level = [0, 20, 40, 60, 80]  # corresponds to qs [0.1, 0.2, ..., 0.9]
    df = generate_series(n_series=2, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=3,
        freq="D",
        level=level,
    )
    exp_lv_cols = []
    for lv in level:
        if lv == 0:
            continue
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df.columns)
    # test monotonicity of levels
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        if model.alias == "ZeroModel":
            # ZeroModel is a constant model, so all levels should be the same
            assert fcst_df[c1].eq(fcst_df[c2]).all()
        elif "chronos" in model.alias.lower():
            # sometimes it gives this condition
            assert fcst_df[c1].le(fcst_df[c2]).all()
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()
