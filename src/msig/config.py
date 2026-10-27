import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
DAY_MS = 86_400_000


def load_experiment(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "experiment.yaml") as f:
        return yaml.safe_load(f)


def day_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def day_start_ms(day: dt.date) -> int:
    return int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp() * 1000)


def latencies(cfg: dict) -> list[int]:
    a = cfg["align"]
    return sorted({a["latency_ms"], *a["latency_sensitivity_ms"]})


def git_hash() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return out + ("-dirty" if dirty else "")


def write_meta(result_path: Path, cfg: dict) -> Path:
    meta_path = result_path.with_name(result_path.stem + ".meta.json")
    meta = {
        "result": result_path.name,
        "git": git_hash(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": cfg,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta_path
