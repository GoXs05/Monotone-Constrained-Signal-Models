import polars as pl

from msig.features.book import feature_mid, l1_imbalance, past_return_bps, spread_ticks
from msig.features.flow import flow_imbalance
from msig.features.regime import realized_vol
from msig.labels import add_labels

LABEL_INPUTS = ["label_mid", "label_ts"]


def feature_columns(cfg: dict) -> list[str]:
    f, lab = cfg["features"], cfg["labels"]
    return [
        "imb_l1",
        "spread_ticks",
        *[f"flow_imb_{w}s" for w in f["flow_lookbacks_s"]],
        f"rv_{f['vol_window_s']}s",
        *[f"past_ret_{h}s" for h in lab["horizons_s"]],
    ]


def label_columns(cfg: dict) -> list[str]:
    hs = cfg["labels"]["horizons_s"]
    cols = [f"y_{h}s" for h in hs]
    if cfg["labels"]["vol_normalize"]:
        cols += [f"yv_{h}s" for h in hs]
    return cols


def build_frame(aligned: pl.DataFrame, cfg: dict, symbol: str) -> pl.DataFrame:
    """Aligned grid (one or more contiguous days) to features and labels."""
    f = cfg["features"]
    df = aligned.sort("t").with_columns(
        feature_mid(), l1_imbalance(), spread_ticks(cfg["tick_size"][symbol])
    )
    df = flow_imbalance(df, f["flow_lookbacks_s"])
    df = realized_vol(df, f["vol_window_s"])
    for h in cfg["labels"]["horizons_s"]:
        df = past_return_bps(df, h)
    df = add_labels(
        df, cfg["labels"]["horizons_s"], cfg["labels"]["vol_normalize"], f"rv_{f['vol_window_s']}s"
    )
    return df.select("t", *feature_columns(cfg), *label_columns(cfg))
