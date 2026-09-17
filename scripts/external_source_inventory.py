#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontiermem.external_data import DATASET_REGISTRY

if __name__ == "__main__":
    print(json.dumps(DATASET_REGISTRY, indent=2, ensure_ascii=False))
