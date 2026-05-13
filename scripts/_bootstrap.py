from __future__ import annotations

from pathlib import Path
import sys


def prepare_script_runtime(script_file: str | Path) -> Path:
    root = Path(script_file).resolve().parents[1]
    for candidate in (root, root / "src"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return root
