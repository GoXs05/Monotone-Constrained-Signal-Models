"""Aligned grid to per-day feature and label parquet, then the zero-label report."""

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
from msig.data.load import load_days, parquet_path
from msig.features.build import build_frame, label_columns

ONE_DAY = dt.timedelta(days=1)


def neighbours(day: dt.date, kind: str, symbol: str, cfg: dict, allow_holdout: bool):
    first = cfg["dev_start"]
    # Labels at the end of the last dev day stay null rather than read holdout mids
    last = cfg["holdout_end"] if allow_holdout else cfg["dev_end"]
    days = [d for d in (day - ONE_DAY, day, day + ONE_DAY) if first <= d <= last]
    missing = [d for d in days if not parquet_path(kind, symbol, d, cfg).exists()]
    if missing:
        raise FileNotFoundError(f"{kind} {symbol} missing {missing}; run align first")
    return days


def build_day(symbol: str, day: dt.date, lat: int, cfg: dict, allow_holdout: bool = False) -> None:
    out = parquet_path(f"features_L{lat}", symbol, day, cfg)
    if out.exists():
        return
    kind = f"aligned_L{lat}"
    aligned = load_days(
        kind, symbol, neighbours(day, kind, symbol, cfg, allow_holdout), cfg, allow_holdout
    )
    start = day_start_ms(day)
    frame = build_frame(aligned, cfg, symbol).filter(
        pl.col("t").is_between(start, start + DAY_MS - 1)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out, compression="zstd")


def zero_label_report(days: list[dt.date], cfg: dict) -> pl.DataFrame:
    lat = cfg["align"]["latency_ms"]
    rows = []
    for symbol in cfg["assets"]:
        paths = [parquet_path(f"features_L{lat}", symbol, d, cfg) for d in days]
        lf = pl.scan_parquet(paths)
        for col in label_columns(cfg):
            r = lf.select(
                n=pl.col(col).is_not_null().sum(),
                zero_frac=(pl.col(col) == 0).sum() / pl.col(col).is_not_null().sum(),
            ).collect()
            rows.append({"symbol": symbol, "label": col, **r.row(0, named=True)})
    return pl.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=dt.date.fromisoformat)
    p.add_argument("--end", type=dt.date.fromisoformat)
    args = p.parse_args()
    cfg = load_experiment()
    lb = [
        *cfg["features"]["flow_lookbacks_s"],
        cfg["features"]["vol_window_s"],
        *cfg["labels"]["horizons_s"],
    ]
    # One neighbour day of context covers every lookback and horizon
    assert max(lb) < 86_400, "lookbacks and horizons must be under one day"

    days = day_range(args.start or cfg["dev_start"], args.end or cfg["dev_end"])
    for symbol in cfg["assets"]:
        for lat in latencies(cfg):
            for day in days:
                build_day(symbol, day, lat, cfg)
            print(f"features {symbol} L={lat} done", flush=True)

    report = zero_label_report(days, cfg)
    out = RESULTS_DIR / "label_zero_fraction.csv"
    report.write_csv(out)
    write_meta(out, cfg)
    print(report)


if __name__ == "__main__":
    main()
