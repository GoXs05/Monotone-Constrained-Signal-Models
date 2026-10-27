"""Stream daily files from data.binance.vision: download, verify, convert, drop raw."""

import argparse
import datetime as dt
import hashlib
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from msig.config import ROOT, day_range, load_experiment
from msig.data.load import check_holdout_access, csv_zip_to_frame, parquet_path

BASE_URL = "https://data.binance.vision/data/futures/{market}/daily/{kind}/{symbol}/{name}"


class ChecksumError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 6) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as e:
            # A 4xx means the file is not there; retrying will not change that
            if e.code < 500 or attempt == retries - 1:
                raise
        except OSError:
            if attempt == retries - 1:
                raise
        time.sleep(min(2**attempt * 5, 300))


def fetch_verified(kind: str, symbol: str, day: dt.date, cfg: dict) -> Path:
    name = f"{symbol}-{kind}-{day.isoformat()}.zip"
    url = BASE_URL.format(market=cfg["data"]["market"], kind=kind, symbol=symbol, name=name)
    raw_dir = ROOT / cfg["data"]["raw_dir"] / kind / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path, sum_path = raw_dir / name, raw_dir / (name + ".CHECKSUM")

    for _ in range(2):
        if not sum_path.exists():
            fetch(url + ".CHECKSUM", sum_path)
        if not zip_path.exists():
            fetch(url, zip_path)
        expected = sum_path.read_text().split()[0].lower()
        if sha256(zip_path) == expected:
            return zip_path
        zip_path.unlink()
        sum_path.unlink()
    raise ChecksumError(f"{name}: sha256 mismatch after retry")


def process_day(
    kind: str, symbol: str, day: dt.date, cfg: dict, allow_holdout: bool = False
) -> Path:
    check_holdout_access([day], cfg, allow_holdout)
    out = parquet_path(kind, symbol, day, cfg)
    if out.exists():
        return out
    zip_path = fetch_verified(kind, symbol, day, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    csv_zip_to_frame(zip_path, kind).write_parquet(tmp, compression="zstd")
    # Rename last so a crash never leaves a parquet that looks complete
    tmp.replace(out)
    if not cfg["data"]["keep_raw"]:
        zip_path.unlink()
        zip_path.with_name(zip_path.name + ".CHECKSUM").unlink()
    return out


def process_range(days: list[dt.date], cfg: dict, allow_holdout: bool = False) -> None:
    for day in days:
        for symbol in cfg["assets"]:
            for kind in cfg["data"]["kinds"]:
                out = process_day(kind, symbol, day, cfg, allow_holdout)
                print(f"{day} {symbol} {kind} -> {out.relative_to(ROOT)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=dt.date.fromisoformat)
    p.add_argument("--end", type=dt.date.fromisoformat)
    args = p.parse_args()
    cfg = load_experiment()
    start = args.start or cfg["dev_start"]
    end = args.end or cfg["dev_end"]
    process_range(day_range(start, end), cfg)


if __name__ == "__main__":
    main()
