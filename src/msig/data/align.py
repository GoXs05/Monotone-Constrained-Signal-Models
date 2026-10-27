"""1s clock grid on the exchange clock with a latency offset for feature inputs.

For grid time t and latency L:
  feature book state = last book row with transaction_time <= t - L
  feature trade bucket = trades with transact_time in (t - L - 1000, t - L]
  label_mid = mid of last book row with transaction_time <= t
"""

import argparse
import datetime as dt

import polars as pl

from msig.config import (
    DAY_MS,
    RESULTS_DIR,
    day_range,
    day_start_ms,
    latencies,
    load_experiment,
    write_meta,
)
from msig.data.load import check_holdout_access, load_days, parquet_path

BOOK_TS = "transaction_time"
TRADE_TS = "transact_time"
GAP_MS = 5_000
# Prior-day context needed for as-of at 00:00:00 and the first trade bucket
CONTEXT_MS = 60_000


def collapse_book(book: pl.DataFrame) -> pl.DataFrame:
    return (
        book.sort(BOOK_TS, "update_id")
        .group_by(BOOK_TS, maintain_order=True)
        .last()
        .select(BOOK_TS, "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty")
    )


def make_grid(start_ms: int, n_seconds: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"t": pl.int_range(start_ms, start_ms + n_seconds * 1000, 1000, eager=True)}
    )


def align_frames(
    book: pl.DataFrame, trades: pl.DataFrame, grid: pl.DataFrame, latency_ms: int
) -> pl.DataFrame:
    book = collapse_book(book)
    feat_book = grid.with_columns(key=pl.col("t") - latency_ms).join_asof(
        book.rename({BOOK_TS: "book_ts"}),
        left_on="key",
        right_on="book_ts",
        strategy="backward",
        allow_exact_matches=True,
    )
    label_book = grid.join_asof(
        book.select(
            pl.col(BOOK_TS).alias("label_ts"),
            ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2).alias("label_mid"),
        ),
        left_on="t",
        right_on="label_ts",
        strategy="backward",
        allow_exact_matches=True,
    )
    # Smallest grid t with ts <= t - L, i.e. ceil((ts + L) / 1000) * 1000
    bucket = -(-(pl.col(TRADE_TS) + latency_ms) // 1000) * 1000
    sign_sell = pl.col("is_buyer_maker")
    flow = (
        trades.with_columns(t=bucket)
        .group_by("t")
        .agg(
            buy_qty=pl.col("quantity").filter(~sign_sell).sum(),
            sell_qty=pl.col("quantity").filter(sign_sell).sum(),
            n_trades=pl.len().cast(pl.Int64),
        )
    )
    return (
        feat_book.drop("key")
        .join(label_book.select("t", "label_ts", "label_mid"), on="t", how="left")
        .join(flow, on="t", how="left")
        .with_columns(pl.col("buy_qty", "sell_qty", "n_trades").fill_null(0))
        .sort("t")
    )


def gap_stats(ts: pl.Series, start_ms: int) -> tuple[int, int]:
    # Day edges count as stamps so an empty morning shows up as a gap
    full = pl.concat([pl.Series([start_ms - 1]), ts, pl.Series([start_ms + DAY_MS])])
    d = full.diff().drop_nulls()
    return int((d > GAP_MS).sum()), int(d.max())


def quality_row(
    book: pl.DataFrame, trades: pl.DataFrame, day: dt.date, symbol: str, lats: list[int]
) -> dict:
    start = day_start_ms(day)
    bts, tts = book[BOOK_TS], trades[TRADE_TS]
    b_gaps, b_max = gap_stats(bts, start)
    t_gaps, t_max = gap_stats(tts, start)
    lag = book["event_time"] - bts
    row = {
        "date": day,
        "symbol": symbol,
        "book_rows": book.height,
        "book_dup_ts_frac": float((bts.diff().drop_nulls() == 0).mean()),
        "book_gaps_gt5s": b_gaps,
        "book_max_gap_ms": b_max,
        "trade_rows": trades.height,
        "trade_dup_ts_frac": float((tts.diff().drop_nulls() == 0).mean()),
        "trade_gaps_gt5s": t_gaps,
        "trade_max_gap_ms": t_max,
    }
    for lat in lats:
        row[f"evt_lag_gt_{lat}ms_frac"] = float((lag > lat).mean())
    return row


def with_context(kind: str, symbol: str, day: dt.date, cfg: dict, allow_holdout: bool):
    ts = BOOK_TS if kind == "bookTicker" else TRADE_TS
    cur = load_days(kind, symbol, [day], cfg, allow_holdout)
    prev_day = day - dt.timedelta(days=1)
    if not parquet_path(kind, symbol, prev_day, cfg).exists():
        return cur, cur
    prev = load_days(kind, symbol, [prev_day], cfg, allow_holdout)
    tail = prev.filter(pl.col(ts) >= day_start_ms(day) - CONTEXT_MS)
    if tail.is_empty():
        tail = prev.tail(1)
    return pl.concat([tail, cur]), cur


def align_day(symbol: str, day: dt.date, cfg: dict, allow_holdout: bool = False) -> dict:
    check_holdout_access([day], cfg, allow_holdout)
    lats = latencies(cfg)
    dq_path = parquet_path("dq", symbol, day, cfg)
    outs = {lat: parquet_path(f"aligned_L{lat}", symbol, day, cfg) for lat in lats}
    if dq_path.exists() and all(p.exists() for p in outs.values()):
        return pl.read_parquet(dq_path).row(0, named=True)

    book, book_day = with_context("bookTicker", symbol, day, cfg, allow_holdout)
    trades, trades_day = with_context("aggTrades", symbol, day, cfg, allow_holdout)
    grid = make_grid(day_start_ms(day), 86_400)
    for lat, out in outs.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        align_frames(book, trades, grid, lat).write_parquet(out, compression="zstd")
    row = quality_row(book_day, trades_day, day, symbol, lats)
    dq_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_parquet(dq_path)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=dt.date.fromisoformat)
    p.add_argument("--end", type=dt.date.fromisoformat)
    args = p.parse_args()
    cfg = load_experiment()
    days = day_range(args.start or cfg["dev_start"], args.end or cfg["dev_end"])
    rows = []
    for symbol in cfg["assets"]:
        for day in days:
            rows.append(align_day(symbol, day, cfg))
            print(f"aligned {symbol} {day}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "data_quality.csv"
    pl.DataFrame(rows).write_csv(out)
    write_meta(out, cfg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
