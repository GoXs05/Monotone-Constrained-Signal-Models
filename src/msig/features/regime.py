import numpy as np
import polars as pl

from msig.data.align import value_at_offset


def realized_vol(df: pl.DataFrame, window_s: int) -> pl.DataFrame:
    """Trailing RMS of 1s feature-mid log returns, bps per sqrt(second); needs `mid_f`."""
    lagged = value_at_offset(df, "mid_f", -1, "_mid_lag")
    r1 = 1e4 * (pl.col("mid_f").log() - pl.col("_mid_lag").log())
    ms = (r1**2).rolling_mean_by("t", f"{window_s * 1000}i", min_samples=window_s // 2)
    # Running-sum rolling mean can dip a hair below zero after a jump leaves the window
    rv = ms.clip(lower_bound=0.0).sqrt()
    return lagged.with_columns(rv.alias(f"rv_{window_s}s")).drop("_mid_lag")


class VolBuckets:
    """Quantile buckets of trailing vol; edges come from training rows only."""

    def __init__(self, n_buckets: int):
        self.n_buckets = n_buckets
        self.edges: np.ndarray | None = None

    def fit(self, vol: np.ndarray) -> "VolBuckets":
        qs = np.arange(1, self.n_buckets) / self.n_buckets
        self.edges = np.nanquantile(vol, qs)
        return self

    def transform(self, vol: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("VolBuckets.transform called before fit")
        out = np.searchsorted(self.edges, vol, side="right")
        return np.where(np.isnan(vol), -1, out)
