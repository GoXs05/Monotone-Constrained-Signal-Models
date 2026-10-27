import polars as pl


def flow_imbalance(df: pl.DataFrame, lookbacks_s: list[int]) -> pl.DataFrame:
    """Signed taker volume imbalance over trailing windows (t - W, t] of trade buckets."""
    cols = []
    for w in lookbacks_s:
        window = f"{w * 1000}i"
        buy = pl.col("buy_qty").rolling_sum_by("t", window)
        sell = pl.col("sell_qty").rolling_sum_by("t", window)
        total = buy + sell
        # No trades in the window reads as zero imbalance, not missing
        imb = pl.when(total > 0).then((buy - sell) / total).otherwise(0.0)
        cols.append(imb.alias(f"flow_imb_{w}s"))
        cols.append(pl.col("n_trades").rolling_sum_by("t", window).alias(f"trades_{w}s"))
    return df.with_columns(cols)
