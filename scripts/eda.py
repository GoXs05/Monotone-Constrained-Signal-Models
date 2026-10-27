"""Binned mean label vs each feature, on the configured EDA window only."""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from msig.config import RESULTS_DIR, day_range, load_experiment, write_meta
from msig.data.load import load_days
from msig.features.build import feature_columns

SERIES = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MAX_DISCRETE = 12


def binned(df: pl.DataFrame, x: str, y: str, n_bins: int) -> pl.DataFrame:
    d = df.select(x, y).drop_nulls()
    if d[x].dtype.is_integer() or d[x].n_unique() <= MAX_DISCRETE:
        key = pl.col(x).cast(pl.Float64)
    else:
        # Unique edges so heavy ties (e.g. zero flow) collapse into one bin
        edges = np.unique(np.quantile(d[x].to_numpy(), np.linspace(0, 1, n_bins + 1)))
        key = pl.col(x).cut(list(edges[1:-1]), left_closed=True).to_physical()
    return (
        d.group_by(key.alias("bin"))
        .agg(
            x=pl.col(x).mean(),
            mean=pl.col(y).mean(),
            se=pl.col(y).std() / pl.len().sqrt(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= 30)
        .sort("x")
    )


def plot(df: pl.DataFrame, feats: list[str], y: str, title: str, n_bins: int):
    ncol = 4
    nrow = math.ceil(len(feats) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow), squeeze=False)
    tables = []
    for ax, f in zip(axes.flat, feats, strict=False):
        b = binned(df, f, y, n_bins)
        tables.append(b.drop("bin").with_columns(feature=pl.lit(f)))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.errorbar(
            b["x"],
            b["mean"],
            yerr=2 * b["se"],
            color=SERIES,
            lw=2,
            marker="o",
            ms=4,
            capsize=0,
            elinewidth=1,
        )
        ax.set_title(f, fontsize=9, color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axes.flat)[len(feats) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.supylabel(f"mean {y} (bps), ±2 SE", fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig, pl.concat(tables)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, action="append", help="seconds; default all")
    p.add_argument("--latency", type=int, help="ms; default align.latency_ms")
    p.add_argument("--bins", type=int, default=20)
    args = p.parse_args()
    cfg = load_experiment()
    start, end = cfg["eda_start"], cfg["eda_end"]
    assert cfg["dev_start"] <= start <= end < cfg["holdout_start"], "EDA window outside dev range"
    days = day_range(start, end)
    lat = args.latency if args.latency is not None else cfg["align"]["latency_ms"]
    horizons = args.horizon or cfg["labels"]["horizons_s"]
    feats = feature_columns(cfg)
    ycol = "yv_{}s" if cfg["labels"]["vol_normalize"] else "y_{}s"

    out_dir = RESULTS_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in cfg["assets"]:
        df = load_days(f"features_L{lat}", symbol, days, cfg)
        for h in horizons:
            # Stride h so each bin's SE counts independent label intervals
            sub = df.gather_every(h)
            y = ycol.format(h)
            title = f"{symbol}  {y}  L={lat}ms  {start} to {end}  (stride {h}s, n={sub.height:,})"
            fig, table = plot(sub, feats, y, title, args.bins)
            stem = out_dir / f"{symbol}_{y}_L{lat}"
            fig.savefig(stem.with_suffix(".png"), dpi=130)
            plt.close(fig)
            table.write_csv(stem.with_suffix(".csv"))
            write_meta(stem.with_suffix(".png"), cfg)
            print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
