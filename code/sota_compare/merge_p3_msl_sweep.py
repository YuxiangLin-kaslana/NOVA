#!/usr/bin/env python3
"""Merge per-entity P3 MSL sweep outputs into the expected collector JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("/u/ylin30/sigLA/code/runs"))
    parser.add_argument("--tag", default="p3_20260708_132402")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pattern = f"sota_multidata_compare_msl_{args.tag}_msl_entity_sweep_*.json"
    files = sorted(args.runs_dir.glob(pattern))
    if not files:
        raise SystemExit(f"no per-entity files matched {pattern}")

    per_entity: dict[str, Any] = {}
    meta = None
    nseed = None
    novel = None
    for path in files:
        data = load(path)
        nseed = data.get("nseed", nseed)
        novel = data.get("novel", novel)
        meta = data.get("meta", meta)
        for ent, recs in data.get("per_entity", {}).items():
            per_entity[ent] = recs

    entities = sorted(per_entity)
    out = args.out or args.runs_dir / f"sota_multidata_compare_msl_{args.tag}_msl_entity_sweep.json"
    payload = {
        "nseed": nseed,
        "dataset": "MSL",
        "entities": entities,
        "novel": novel,
        "per_entity": per_entity,
        "meta": meta,
        "merged_from": [p.name for p in files],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"merged {len(files)} files, {len(entities)} entities -> {out}")


if __name__ == "__main__":
    main()
