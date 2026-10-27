import zipfile

import pytest

from msig.config import load_experiment
from msig.data.load import (
    SCHEMAS,
    HoldoutAccessError,
    check_holdout_access,
    csv_zip_to_frame,
    load_days,
)

CFG = load_experiment()


def test_holdout_date_raises_without_flag():
    with pytest.raises(HoldoutAccessError):
        load_days("bookTicker", "BTCUSDT", [CFG["holdout_start"]], CFG)


def test_mixed_range_raises():
    days = [CFG["dev_end"], CFG["holdout_start"]]
    with pytest.raises(HoldoutAccessError):
        check_holdout_access(days, CFG)


def test_flag_allows_holdout():
    check_holdout_access([CFG["holdout_start"]], CFG, allow_holdout=True)


def test_dev_dates_pass():
    check_holdout_access([CFG["dev_start"], CFG["dev_end"]], CFG)


def _zip(tmp_path, text):
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.csv", text)
    return path


@pytest.mark.parametrize("header", [True, False])
def test_book_csv_with_and_without_header(tmp_path, header):
    body = "1,100.0,2.0,100.1,3.0,1685577600009,1685577600014\n"
    hdr = ",".join(SCHEMAS["bookTicker"]) + "\n"
    df = csv_zip_to_frame(_zip(tmp_path, (hdr if header else "") + body), "bookTicker")
    assert df.height == 1
    assert df["transaction_time"][0] == 1685577600009
    assert df["event_time"][0] == 1685577600014


def test_trade_csv_bool_parse(tmp_path):
    body = ",".join(SCHEMAS["aggTrades"]) + "\n1,100.0,0.5,1,1,1685577600033,true\n"
    df = csv_zip_to_frame(_zip(tmp_path, body), "aggTrades")
    assert df["is_buyer_maker"][0] is True
