import numpy as np
import polars as pl
import pytest

from msig.data.align import align_frames, make_grid
from msig.data.load import SCHEMAS
from msig.features.build import build_frame, model_features

T0 = 1_700_000_000_000
N = 120
CFG = {
    "tick_size": {"BTCUSDT": 0.1},
    "features": {"model": ["imb_l1"], "flow_lookbacks_s": [5], "vol_window_s": 30},
    "labels": {"horizons_s": [1, 5, 30], "vol_normalize": False},
}


def mid_asof(ts, mids, at):
    return mids[np.searchsorted(ts, at, side="right") - 1]


@pytest.mark.parametrize("lat", [0, 50, 250])
def test_past_return_anchored_at_t_minus_latency(lat):
    rng = np.random.default_rng(lat)
    ts = np.arange(T0 - 60_000, T0 + N * 1000, 7, dtype=np.int64)
    mids = 30_000 + np.cumsum(rng.choice([-0.1, 0.0, 0.1], ts.size))
    book = pl.DataFrame(
        {
            "update_id": np.arange(ts.size, dtype=np.int64),
            "best_bid_price": mids - 0.05,
            "best_bid_qty": np.ones(ts.size),
            "best_ask_price": mids + 0.05,
            "best_ask_qty": np.ones(ts.size),
            "transaction_time": ts,
            "event_time": ts,
        }
    )
    trades = pl.DataFrame(schema=SCHEMAS["aggTrades"])
    out = build_frame(align_frames(book, trades, make_grid(T0, N), lat), CFG, "BTCUSDT")
    t = out["t"].to_numpy()
    for h in CFG["labels"]["horizons_s"]:
        want = 1e4 * np.log(mid_asof(ts, mids, t - lat) / mid_asof(ts, mids, t - lat - h * 1000))
        got = out[f"past_ret_{h}s"].to_numpy()
        np.testing.assert_allclose(got[h:], want[h:], rtol=0, atol=1e-9)
        if lat:
            at_t = 1e4 * np.log(mid_asof(ts, mids, t) / mid_asof(ts, mids, t - h * 1000))
            assert not np.allclose(got[h:], at_t[h:])


def test_spread_cannot_be_model_input():
    cfg = {**CFG, "features": {**CFG["features"], "model": ["imb_l1", "spread_ticks"]}}
    with pytest.raises(ValueError):
        model_features(cfg)
