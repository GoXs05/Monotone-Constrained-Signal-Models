import datetime as dt
import importlib.util

import numpy as np
import polars as pl
import pytest

import msig.data.align as align
from msig.config import ROOT, day_start_ms
from msig.data.load import parquet_path
from msig.features.build import feature_stamp
from msig.provenance import is_current, make_stamp, read_stamp, write_stamped

DAY = dt.date(2023, 6, 1)


def load_script():
    spec = importlib.util.spec_from_file_location("bf", ROOT / "scripts" / "build_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg(tmp_path):
    return {
        "dev_start": DAY,
        "dev_end": DAY,
        "holdout_start": DAY + dt.timedelta(days=30),
        "holdout_end": DAY + dt.timedelta(days=40),
        "data": {"parquet_dir": str(tmp_path)},
        "align": {"latency_ms": 50, "latency_sensitivity_ms": [0, 50]},
        "tick_size": {"BTCUSDT": 0.1},
        "features": {"model": ["imb_l1"], "flow_lookbacks_s": [5], "vol_window_s": 30},
        "labels": {"horizons_s": [1, 5], "vol_normalize": False},
    }


@pytest.fixture
def raw_day(cfg):
    rng = np.random.default_rng(0)
    start = day_start_ms(DAY)
    ts = np.sort(rng.integers(start, start + 86_400_000, 50_000))
    mid = 30_000 + np.cumsum(rng.choice([-0.1, 0.0, 0.1], ts.size))
    book = pl.DataFrame(
        {
            "update_id": np.arange(ts.size, dtype=np.int64),
            "best_bid_price": mid - 0.05,
            "best_bid_qty": np.ones(ts.size),
            "best_ask_price": mid + 0.05,
            "best_ask_qty": np.ones(ts.size),
            "transaction_time": ts,
            "event_time": ts + 5,
        }
    )
    tts = np.sort(rng.integers(start, start + 86_400_000, 5_000))
    trades = pl.DataFrame(
        {
            "agg_trade_id": np.arange(tts.size, dtype=np.int64),
            "price": np.full(tts.size, 30_000.0),
            "quantity": np.ones(tts.size),
            "first_trade_id": np.arange(tts.size, dtype=np.int64),
            "last_trade_id": np.arange(tts.size, dtype=np.int64),
            "transact_time": tts,
            "is_buyer_maker": rng.random(tts.size) < 0.5,
        }
    )
    for kind, df in (("bookTicker", book), ("aggTrades", trades)):
        path = parquet_path(kind, "BTCUSDT", DAY, cfg)
        path.parent.mkdir(parents=True)
        df.write_parquet(path)


def test_stamp_roundtrip_and_match(tmp_path):
    path = tmp_path / "x.parquet"
    a = make_stamp({"k": 1}, ["src/msig/provenance.py"], "abc")
    assert not is_current(path, a)
    pl.DataFrame({"x": [1]}).write_parquet(path)
    assert not is_current(path, a)  # unstamped file counts as stale
    write_stamped(pl.DataFrame({"x": [1]}), path, a)
    assert read_stamp(path) == a
    assert is_current(path, a)
    assert is_current(path, {**a, "git": "other"})  # git alone is provenance, not staleness
    assert not is_current(path, make_stamp({"k": 2}, ["src/msig/provenance.py"], "abc"))
    assert not is_current(path, {**a, "code_hash": "0" * 16})


def test_stale_align_stamp_triggers_rebuild(cfg, raw_day, monkeypatch):
    calls = []
    real = align.align_frames
    monkeypatch.setattr(align, "align_frames", lambda *a: calls.append(a[-1]) or real(*a))

    stamps = align.align_stamps(cfg, "g1")
    align.align_day("BTCUSDT", DAY, cfg, stamps)
    assert sorted(calls) == [0, 50]

    calls.clear()
    align.align_day("BTCUSDT", DAY, cfg, stamps)
    assert calls == []

    stale = {**stamps, 50: {**stamps[50], "config_hash": "stale"}}
    path = parquet_path("aligned_L50", "BTCUSDT", DAY, cfg)
    write_stamped(pl.read_parquet(path), path, stale[50])
    align.align_day("BTCUSDT", DAY, cfg, stamps)
    assert calls == [50]
    assert read_stamp(path)["config_hash"] == stamps[50]["config_hash"]


def test_stale_feature_stamp_triggers_rebuild(cfg, raw_day):
    bf = load_script()
    align.align_day("BTCUSDT", DAY, cfg, align.align_stamps(cfg, "g1"))
    stamp = feature_stamp(cfg, "BTCUSDT", 50, "g1")
    assert bf.build_day("BTCUSDT", DAY, 50, cfg, stamp) is True
    assert bf.build_day("BTCUSDT", DAY, 50, cfg, stamp) is False

    cfg["features"]["vol_window_s"] = 60
    new = feature_stamp(cfg, "BTCUSDT", 50, "g1")
    assert new["config_hash"] != stamp["config_hash"]
    assert bf.build_day("BTCUSDT", DAY, 50, cfg, new) is True
    out = parquet_path("features_L50", "BTCUSDT", DAY, cfg)
    assert read_stamp(out)["config_hash"] == new["config_hash"]
    assert "rv_60s" in pl.read_parquet(out).columns


def test_dirty_tree_refused_unless_allowed(monkeypatch):
    import subprocess

    import msig.provenance as prov

    class Out:
        stdout = " M src/msig/labels.py\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Out())
    with pytest.raises(prov.DirtyTreeError):
        prov.require_clean(allow_dirty=False)
    prov.require_clean(allow_dirty=True)


def test_lockfile_change_makes_stamp_stale(tmp_path):
    path = tmp_path / "x.parquet"
    a = make_stamp({"k": 1}, ["src/msig/provenance.py"], "abc")
    assert "lock_hash" in a
    write_stamped(pl.DataFrame({"x": [1]}), path, a)
    assert not is_current(path, {**a, "lock_hash": "0" * 16})
