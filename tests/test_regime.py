import numpy as np
import polars as pl
import pytest

from msig.features.regime import VolBuckets, realized_vol


def test_edges_come_from_fit_data_only():
    train = np.arange(1.0, 101.0)
    vb = VolBuckets(4).fit(train)
    edges = vb.edges.copy()
    vb.transform(np.full(10, 1e6))
    np.testing.assert_array_equal(vb.edges, edges)
    np.testing.assert_array_equal(vb.transform(np.array([0.0, 30.0, 60.0, 1e6])), [0, 1, 2, 3])


def test_nan_maps_to_minus_one():
    vb = VolBuckets(3).fit(np.array([1.0, 2.0, 3.0, np.nan]))
    assert vb.transform(np.array([np.nan]))[0] == -1


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        VolBuckets(3).transform(np.array([1.0]))


def test_realized_vol_never_nan_in_quiet_stretch():
    # Seed 9 drives polars' running-sum rolling mean to about -4e-19 in a flat stretch
    rng = np.random.default_rng(9)
    n = 20_000
    steps = rng.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1]) * 0.05 * rng.integers(1, 30, n)
    steps[(np.arange(n) // 1500) % 2 == 1] = 0.0
    df = pl.DataFrame({"t": np.arange(n) * 1000, "mid_f": 27_000 + np.cumsum(steps)})
    rv = realized_vol(df, 300)["rv_300s"].drop_nulls()
    assert not rv.is_nan().any()
    assert (rv >= 0).all()
