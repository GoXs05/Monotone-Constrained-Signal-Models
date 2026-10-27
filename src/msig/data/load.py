"""Raw csv zip to parquet, and the guarded parquet reader used by everything downstream."""

import datetime as dt
import io
import zipfile
from pathlib import Path

import polars as pl

from msig.config import ROOT, load_experiment

SCHEMAS = {
    "bookTicker": {
        "update_id": pl.Int64,
        "best_bid_price": pl.Float64,
        "best_bid_qty": pl.Float64,
        "best_ask_price": pl.Float64,
        "best_ask_qty": pl.Float64,
        "transaction_time": pl.Int64,
        "event_time": pl.Int64,
    },
    "aggTrades": {
        "agg_trade_id": pl.Int64,
        "price": pl.Float64,
        "quantity": pl.Float64,
        "first_trade_id": pl.Int64,
        "last_trade_id": pl.Int64,
        "transact_time": pl.Int64,
        "is_buyer_maker": pl.Boolean,
    },
}


class HoldoutAccessError(RuntimeError):
    pass


def check_holdout_access(dates: list[dt.date], cfg: dict, allow_holdout: bool = False) -> None:
    if allow_holdout:
        return
    hit = [d for d in dates if d >= cfg["holdout_start"]]
    if hit:
        raise HoldoutAccessError(
            f"{len(hit)} holdout date(s) from {min(hit)} requested without allow_holdout"
        )


def csv_zip_to_frame(zip_path: Path, kind: str) -> pl.DataFrame:
    schema = SCHEMAS[kind]
    with zipfile.ZipFile(zip_path) as zf:
        (name,) = zf.namelist()
        raw = zf.read(name)
    # Older Binance files have no header row
    has_header = not raw[:1].isdigit()
    return pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        new_columns=list(schema),
        schema_overrides=schema,
    ).select(list(schema))


def parquet_path(kind: str, symbol: str, day: dt.date, cfg: dict) -> Path:
    return ROOT / cfg["data"]["parquet_dir"] / kind / symbol / f"{day.isoformat()}.parquet"


def load_days(
    kind: str,
    symbol: str,
    days: list[dt.date],
    cfg: dict | None = None,
    allow_holdout: bool = False,
) -> pl.DataFrame:
    cfg = cfg or load_experiment()
    check_holdout_access(days, cfg, allow_holdout)
    return pl.concat([pl.read_parquet(parquet_path(kind, symbol, d, cfg)) for d in days])
