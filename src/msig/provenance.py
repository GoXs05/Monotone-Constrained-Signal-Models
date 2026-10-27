"""Stamps in parquet metadata so derived files rebuild when their code or config changes."""

import hashlib
import json
import subprocess
from functools import cache
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from msig.config import ROOT

STAMP_KEY = b"msig_stamp"
# Fields that decide staleness; git is recorded for provenance only
MATCH_FIELDS = ("code_hash", "config_hash", "lock_hash")
LOCKFILE = "uv.lock"


class DirtyTreeError(RuntimeError):
    pass


def require_clean(allow_dirty: bool) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if status and not allow_dirty:
        raise DirtyTreeError(f"working tree is dirty, commit or pass --allow-dirty:\n{status}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _read(path: str) -> bytes:
    # Normalize line endings so Windows checkouts hash the same as the committed blobs
    return (ROOT / path).read_bytes().replace(b"\r\n", b"\n")


@cache
def code_hash(paths: tuple[str, ...]) -> str:
    return _sha(b"\0".join(sorted(p.encode() + b"\0" + _read(p) for p in paths)))


@cache
def lock_hash() -> str:
    return _sha(_read(LOCKFILE))


def config_hash(payload: dict) -> str:
    return _sha(json.dumps(payload, sort_keys=True, default=str).encode())


def make_stamp(payload: dict, code_paths: list[str], git: str) -> dict:
    return {
        "git": git,
        "code_hash": code_hash(tuple(code_paths)),
        "lock_hash": lock_hash(),
        "config_hash": config_hash(payload),
        "config": json.loads(json.dumps(payload, sort_keys=True, default=str)),
    }


def write_stamped(df: pl.DataFrame, path: Path, stamp: dict) -> None:
    table = df.to_arrow()
    meta = {**(table.schema.metadata or {}), STAMP_KEY: json.dumps(stamp).encode()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pq.write_table(table.replace_schema_metadata(meta), tmp, compression="zstd")
    # Rename last so a crash never leaves a file that looks complete
    tmp.replace(path)


def read_stamp(path: Path) -> dict | None:
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(STAMP_KEY)
    return json.loads(raw) if raw else None


def is_current(path: Path, stamp: dict) -> bool:
    if not path.exists():
        return False
    old = read_stamp(path)
    return old is not None and all(old.get(k) == stamp[k] for k in MATCH_FIELDS)
