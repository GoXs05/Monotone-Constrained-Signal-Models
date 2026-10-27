import math

import polars as pl
import pytest

from msig.labels import add_labels, forward_return_bps

T0 = 1_700_000_000_000
MIDS = [100.0, 101.0, 99.0, 102.0, 102.0, 98.0, 100.0, 103.0]


def grid(mids, skip=()):
    rows = [(T0 + 1000 * i, m) for i, m in enumerate(mids) if i not in skip]
    return pl.DataFrame({"t": [r[0] for r in rows], "label_mid": [r[1] for r in rows]})


@pytest.mark.parametrize("h", [1, 2, 3])
def test_label_uses_mid_at_t_and_t_plus_h(h):
    y = forward_return_bps(grid(MIDS), h)[f"y_{h}s"].to_list()
    for i in range(len(MIDS) - h):
        assert y[i] == pytest.approx(1e4 * math.log(MIDS[i + h] / MIDS[i]))
    assert y[len(MIDS) - h :] == [None] * h


def test_label_ignores_other_mids():
    h, i = 2, 1
    base = forward_return_bps(grid(MIDS), h)[f"y_{h}s"][i]
    bumped = [m if j in (i, i + h) else m * 7 for j, m in enumerate(MIDS)]
    assert forward_return_bps(grid(bumped), h)[f"y_{h}s"][i] == pytest.approx(base)


def test_exact_zero_when_mid_unchanged():
    assert forward_return_bps(grid(MIDS), 1)["y_1s"][3] == 0.0


def test_missing_grid_row_gives_null_not_shift():
    df = forward_return_bps(grid(MIDS, skip={3}), 2)
    row = df.filter(pl.col("t") == T0 + 1000)
    assert row["y_2s"][0] is None


def test_vol_normalized_scales_by_sqrt_h():
    df = grid(MIDS).with_columns(rv=pl.lit(2.0))
    out = add_labels(df, [4], vol_normalize=True, vol_col="rv")
    assert out["yv_4s"][0] == pytest.approx(out["y_4s"][0] / (2.0 * 2.0))
