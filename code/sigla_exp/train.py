from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.train.cli import main, parse_args


__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
