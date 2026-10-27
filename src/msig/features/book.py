import polars as pl

from msig.data.align import value_at_offset


def feature_mid() -> pl.Expr:
    return ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2).alias("mid_f")


def l1_imbalance() -> pl.Expr:
    bid, ask = pl.col("best_bid_qty"), pl.col("best_ask_qty")
    return ((bid - ask) / (bid + ask)).alias("imb_l1")


def spread_ticks(tick: float) -> pl.Expr:
    spread = (pl.col("best_ask_price") - pl.col("best_bid_price")) / tick
    return spread.round(0).cast(pl.Int32).alias("spread_ticks")


def past_return_bps(df: pl.DataFrame, h: int) -> pl.DataFrame:
    """Log return of the feature mid over the trailing h seconds; needs `mid_f`."""
    lagged = value_at_offset(df, "mid_f", -h, "_mid_lag")
    ret = 1e4 * (pl.col("mid_f").log() - pl.col("_mid_lag").log())
    return lagged.with_columns(ret.alias(f"past_ret_{h}s")).drop("_mid_lag")
