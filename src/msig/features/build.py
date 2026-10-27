import polars as pl

from msig.features.book import feature_mid, l1_imbalance, past_return_bps, spread_ticks
from msig.features.flow import flow_imbalance
from msig.features.regime import realized_vol
from msig.labels import add_labels

LABEL_INPUTS = ["label_mid", "label_ts"]


# Book as-of t for execution cost only, never an entry input, so the lookahead test exempts it
EXECUTION_ONLY = ["exec_spread_ticks"]


def model_features(cfg: dict) -> list[str]:
    feats = list(cfg["features"]["model"])
    banned = set(feats) & set(EXECUTION_ONLY)
    if banned:
        raise ValueError(f"execution-only columns cannot be model inputs: {sorted(banned)}")
    return feats


def aux_columns(cfg: dict) -> list[str]:
    """Known-at-t columns that are not model inputs: costs, regimes, baselines, data quality."""
    f, lab = cfg["features"], cfg["labels"]
    return [
        "spread_ticks",
        f"rv_{f['vol_window_s']}s",
        *[f"past_ret_{h}s" for h in lab["horizons_s"]],
        *[f"trades_{w}s" for w in f["flow_lookbacks_s"]],
    ]


def feature_columns(cfg: dict) -> list[str]:
    computed = [
        "imb_l1",
        *[f"flow_imb_{w}s" for w in cfg["features"]["flow_lookbacks_s"]],
        *aux_columns(cfg),
    ]
    missing = set(model_features(cfg)) - set(computed)
    if missing:
        raise ValueError(f"model features not computed: {sorted(missing)}")
    return computed


def label_columns(cfg: dict) -> list[str]:
    hs = cfg["labels"]["horizons_s"]
    cols = [f"y_{h}s" for h in hs]
    if cfg["labels"]["vol_normalize"]:
        cols += [f"yv_{h}s" for h in hs]
    return cols


def build_frame(aligned: pl.DataFrame, cfg: dict, symbol: str) -> pl.DataFrame:
    """Aligned grid (one or more contiguous days) to features and labels."""
    f = cfg["features"]
    tick = cfg["tick_size"][symbol]
    df = aligned.sort("t").with_columns(
        feature_mid(),
        l1_imbalance(),
        spread_ticks(tick),
        (pl.col("exec_spread_px") / tick).round(0).cast(pl.Int32).alias("exec_spread_ticks"),
    )
    df = flow_imbalance(df, f["flow_lookbacks_s"])
    df = realized_vol(df, f["vol_window_s"])
    for h in cfg["labels"]["horizons_s"]:
        df = past_return_bps(df, h)
    df = add_labels(
        df, cfg["labels"]["horizons_s"], cfg["labels"]["vol_normalize"], f"rv_{f['vol_window_s']}s"
    )
    return df.select("t", *feature_columns(cfg), *EXECUTION_ONLY, *label_columns(cfg))
