import polars as pl

from msig.data.align import value_at_offset


def forward_return_bps(df: pl.DataFrame, h: int) -> pl.DataFrame:
    """Log return of `label_mid` from t to t + h seconds, in bps; null when t + h is off-grid."""
    ahead = value_at_offset(df, "label_mid", h, "_mid_end")
    ret = 1e4 * (pl.col("_mid_end").log() - pl.col("label_mid").log())
    return ahead.with_columns(ret.alias(f"y_{h}s")).drop("_mid_end")


def add_labels(
    df: pl.DataFrame, horizons_s: list[int], vol_normalize: bool, vol_col: str
) -> pl.DataFrame:
    for h in horizons_s:
        df = forward_return_bps(df, h)
        if vol_normalize:
            # Scale trailing per-second vol to the horizon so labels are unitless
            yv = pl.col(f"y_{h}s") / (pl.col(vol_col) * h**0.5)
            df = df.with_columns(yv.alias(f"yv_{h}s"))
    return df
