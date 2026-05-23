#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "plotting" / "03_make_event_study_figures.py"
    runpy.run_path(str(target), run_name="__main__")
