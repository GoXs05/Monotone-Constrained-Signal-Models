"""Replace raw data after the feature cutoff with noise; features at or before t must not move."""

import numpy as np
import polars as pl
import pytest

from msig.data.align import align_frames, make_grid
from msig.features.build import build_frame, feature_columns

T0 = 1_700_000_000_000
N_SECONDS = 900
SYMBOL = "BTCUSDT"
CFG = {
    "tick_size": {SYMBOL: 0.1},
    "features": {
        "model": ["imb_l1", "flow_imb_5s", "flow_imb_30s"],
        "flow_lookbacks_s": [1, 5, 30],
        "vol_window_s": 60,
    },
    "labels": {"horizons_s": [1, 5, 30], "vol_normalize": True},
}


def raw_book(rng, start, end, n):
    ts = np.sort(rng.integers(start, end, n))
    mid = 30_000 + np.cumsum(rng.choice([-0.1, 0.0, 0.1], n))
    half = 0.05 * rng.integers(1, 4, n)
    return pl.DataFrame(
        {
            "update_id": np.arange(n, dtype=np.int64),
            "best_bid_price": mid - half,
            "best_bid_qty": rng.exponential(5, n),
            "best_ask_price": mid + half,
            "best_ask_qty": rng.exponential(5, n),
            "transaction_time": ts,
            "event_time": ts + rng.integers(0, 30, n),
        }
    )


def raw_trades(rng, start, end, n):
    return pl.DataFrame(
        {
            "agg_trade_id": np.arange(n, dtype=np.int64),
            "price": 30_000 + rng.normal(0, 1, n),
            "quantity": rng.exponential(0.5, n),
            "first_trade_id": np.arange(n, dtype=np.int64),
            "last_trade_id": np.arange(n, dtype=np.int64),
            "transact_time": np.sort(rng.integers(start, end, n)),
            "is_buyer_maker": rng.random(n) < 0.5,
        }
    )


def corrupt_after(df, ts_col, cutoff, noise):
    """Drop every row stamped after cutoff and splice in unrelated noise rows."""
    kept = df.filter(pl.col(ts_col) <= cutoff)
    fresh = noise.filter(pl.col(ts_col) > cutoff)
    if "update_id" in df.columns:
        fresh = fresh.with_columns(pl.col("update_id") + 10**9)
    return pl.concat([kept, fresh]).sort(ts_col)


def features(book, trades, lat):
    aligned = align_frames(book, trades, make_grid(T0, N_SECONDS), lat)
    return build_frame(aligned, CFG, SYMBOL).select("t", *feature_columns(CFG))


@pytest.mark.parametrize("lat", [0, 50, 250])
def test_features_ignore_data_after_cutoff(lat):
    rng = np.random.default_rng(lat)
    span = (T0 - 120_000, T0 + N_SECONDS * 1000)
    book, trades = raw_book(rng, *span, 40_000), raw_trades(rng, *span, 8_000)
    base = features(book, trades, lat)

    for t in T0 + 1000 * rng.integers(1, N_SECONDS - 1, 20):
        cutoff = int(t) - lat
        noisy_book = corrupt_after(book, "transaction_time", cutoff, raw_book(rng, *span, 40_000))
        noisy_trades = corrupt_after(trades, "transact_time", cutoff, raw_trades(rng, *span, 8_000))
        new = features(noisy_book, noisy_trades, lat)
        before = pl.col("t") <= t
        assert base.filter(before).equals(new.filter(before)), f"lookahead at t={t}, L={lat}"
        # The corruption has to bite somewhere or the check is vacuous
        assert not base.filter(~before).equals(new.filter(~before))
