import polars as pl
import pytest

from msig.data.align import align_frames, make_grid
from msig.data.load import SCHEMAS

T0 = 1_000_000_000_000
L = 50


def book(rows):
    """rows: (transaction_time, update_id, bid, ask)"""
    return pl.DataFrame(
        {
            "update_id": [r[1] for r in rows],
            "best_bid_price": [float(r[2]) for r in rows],
            "best_bid_qty": [1.0] * len(rows),
            "best_ask_price": [float(r[3]) for r in rows],
            "best_ask_qty": [1.0] * len(rows),
            "transaction_time": [r[0] for r in rows],
            "event_time": [r[0] + 5 for r in rows],
        }
    )


def trades(rows):
    """rows: (transact_time, qty, is_buyer_maker)"""
    return pl.DataFrame(
        {
            "agg_trade_id": list(range(len(rows))),
            "price": [100.0] * len(rows),
            "quantity": [float(r[1]) for r in rows],
            "first_trade_id": list(range(len(rows))),
            "last_trade_id": list(range(len(rows))),
            "transact_time": [r[0] for r in rows],
            "is_buyer_maker": [r[2] for r in rows],
        },
        schema_overrides={"transact_time": pl.Int64},
    )


NO_TRADES = pl.DataFrame(schema=SCHEMAS["aggTrades"])


@pytest.fixture
def fixture_book():
    # Grid points at T0, T0+1000, T0+2000
    return book(
        [
            (T0 - 500, 1, 99, 101),
            (T0 - L, 2, 100, 102),  # exactly t - L for t = T0: visible
            (T0 - L + 1, 3, 101, 103),  # one ms late for t = T0: hidden
            (T0 + 400, 4, 102, 104),
            (T0 + 400, 5, 103, 105),  # same ms, higher update_id wins
            (T0 + 1000, 6, 104, 106),  # stamped at t: label sees it, features do not
            (T0 + 2500, 7, 105, 107),  # after the last grid point
        ]
    )


def test_asof_never_after_cutoff(fixture_book):
    for lat in (0, L, 250):
        out = align_frames(fixture_book, NO_TRADES, make_grid(T0, 3), lat)
        assert (out["book_ts"] <= out["t"] - lat).all()
        assert (out["label_ts"] <= out["t"]).all()


def test_latency_boundary(fixture_book):
    out = align_frames(fixture_book, NO_TRADES, make_grid(T0, 3), L)
    first = out.row(0, named=True)
    assert first["book_ts"] == T0 - L
    assert first["best_bid_price"] == 100.0


def test_label_mid_asof_t(fixture_book):
    out = align_frames(fixture_book, NO_TRADES, make_grid(T0, 3), L)
    assert out["label_mid"].to_list() == [102.0, 105.0, 105.0]
    # Feature at T0+1000 sees the T0+400 row, not the one stamped at T0+1000
    assert out["best_bid_price"].to_list() == [100.0, 103.0, 104.0]


def test_same_ms_collapse_keeps_last_update_id(fixture_book):
    shuffled = fixture_book.sample(fraction=1.0, shuffle=True, seed=0)
    out = align_frames(shuffled, NO_TRADES, make_grid(T0, 3), L)
    assert out.row(1, named=True)["best_bid_price"] == 103.0


def test_trade_bucket_boundaries():
    b = book([(T0 - 10_000, 1, 99, 101)])
    tr = trades(
        [
            (T0 - L, 1.0, False),  # exactly t - L: in bucket t = T0
            (T0 - L - 999, 2.0, True),  # just inside (t - L - 1000, t - L]
            (T0 - L - 1000, 4.0, False),  # left edge excluded, falls in previous bucket
            (T0 - L + 1, 8.0, False),  # one ms late: next bucket
        ]
    )
    out = align_frames(b, tr, make_grid(T0, 2), L)
    r0, r1 = out.row(0, named=True), out.row(1, named=True)
    assert (r0["buy_qty"], r0["sell_qty"], r0["n_trades"]) == (1.0, 2.0, 2)
    assert (r1["buy_qty"], r1["sell_qty"], r1["n_trades"]) == (8.0, 0.0, 1)
