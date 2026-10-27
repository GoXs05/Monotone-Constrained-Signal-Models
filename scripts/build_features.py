"""Aligned grid to per-day feature and label parquet, then the zero-label report."""

import argparse
import datetime as dt

import polars as pl

from msig.config import (
    DAY_MS,
    RESULTS_DIR,
    day_range,
    day_start_ms,
    git_hash,
    latencies,
    load_experiment,
    write_meta,
)
from msig.data.load import load_days, parquet_path
from msig.features.build import build_frame, feature_stamp, label_columns
from msig.provenance import is_current, require_clean, write_stamped

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


def build_day(
    symbol: str, day: dt.date, lat: int, cfg: dict, stamp: dict, allow_holdout: bool = False
) -> bool:
    """Returns True when the file was (re)built."""
    out = parquet_path(f"features_L{lat}", symbol, day, cfg)
    if is_current(out, stamp):
        return False
    kind = f"aligned_L{lat}"
    aligned = load_days(
        kind, symbol, neighbours(day, kind, symbol, cfg, allow_holdout), cfg, allow_holdout
    )
    start = day_start_ms(day)
    frame = build_frame(aligned, cfg, symbol).filter(
        pl.col("t").is_between(start, start + DAY_MS - 1)
    )
    write_stamped(frame, out, stamp)
    return True


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


def empty_window_report(days: list[dt.date], cfg: dict) -> pl.DataFrame:
    """Per day, fraction of grid points whose trailing trade window holds no trades."""
    lat = cfg["align"]["latency_ms"]
    frames = []
    for symbol in cfg["assets"]:
        paths = [parquet_path(f"features_L{lat}", symbol, d, cfg) for d in days]
        frames.append(
            pl.scan_parquet(paths)
            .group_by(date=pl.from_epoch("t", time_unit="ms").dt.date())
            .agg(
                (pl.col(f"trades_{w}s") == 0).mean().alias(f"empty_{w}s_frac")
                for w in cfg["features"]["flow_lookbacks_s"]
            )
            .with_columns(symbol=pl.lit(symbol))
            .collect()
        )
    return pl.concat(frames)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=dt.date.fromisoformat)
    p.add_argument("--end", type=dt.date.fromisoformat)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    require_clean(args.allow_dirty)
    git = git_hash()
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
            stamp = feature_stamp(cfg, symbol, lat, git)
            built = sum(build_day(symbol, day, lat, cfg, stamp) for day in days)
            print(
                f"features {symbol} L={lat}: built {built}, current {len(days) - built}", flush=True
            )

    dq_path = RESULTS_DIR / "data_quality.csv"
    empty = empty_window_report(days, cfg)
    dq = pl.read_csv(dq_path, try_parse_dates=True)
    dq = dq.drop([c for c in empty.columns if c.startswith("empty_")], strict=False)
    dq.join(empty, on=["date", "symbol"], how="left").write_csv(dq_path)
    write_meta(dq_path, cfg)

    report = zero_label_report(days, cfg)
    out = RESULTS_DIR / "label_zero_fraction.csv"
    report.write_csv(out)
    write_meta(out, cfg)
    print(report)


if __name__ == "__main__":
    main()
