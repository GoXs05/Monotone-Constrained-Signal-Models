"""Every msig module an entry point loads must be in that step's hashed source set."""

import json
import subprocess
import sys

import pytest

from msig.config import ROOT
from msig.data.align import ALIGN_CODE
from msig.features.build import FEATURE_CODE

PROBE = """
import importlib.util, json, sys
from pathlib import Path
target = sys.argv[1]
if target.endswith(".py"):
    spec = importlib.util.spec_from_file_location("entry", target)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
else:
    importlib.import_module(target)
root = Path(sys.argv[2])
files = [Path(m.__file__) for n, m in sys.modules.items()
         if (n == "msig" or n.startswith("msig.")) and getattr(m, "__file__", None)]
print(json.dumps(sorted(p.resolve().relative_to(root).as_posix() for p in files)))
"""


def loaded_sources(target: str) -> set[str]:
    # Fresh interpreter so modules imported by other tests do not leak in
    out = subprocess.run(
        [sys.executable, "-c", PROBE, target, str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(json.loads(out))


@pytest.mark.parametrize(
    "target, hashed, entry_file",
    [
        ("msig.data.align", ALIGN_CODE, "src/msig/data/align.py"),
        (str(ROOT / "scripts" / "build_features.py"), FEATURE_CODE, "scripts/build_features.py"),
    ],
)
def test_loaded_modules_are_hashed(target, hashed, entry_file):
    loaded = loaded_sources(target)
    assert loaded, "probe found no msig modules"
    missing = loaded - set(hashed)
    assert not missing, f"loaded but not hashed: {sorted(missing)}"
    assert entry_file in hashed
