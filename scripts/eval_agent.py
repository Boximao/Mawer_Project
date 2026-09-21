"""Citation-hit eval (JSONL). Same as `python src/eval/run_eval.py`."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "eval" / "run_eval.py"), run_name="__main__")
