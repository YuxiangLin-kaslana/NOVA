#!/usr/bin/env python3
"""Reclassify archived naming runs under explicit Level 1/2/3 definitions.

This analysis does not make new API calls. It audits the archived live-LLM
free-form run, whose prompt listed only known concepts and asked the model to
invent a short name for unmatched evidence. The old artifact stores aggregate
rates, not raw generations, so human re-rating is impossible and the result is
reported as an automatic-matching pilot only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "code" / "runs"
DEFAULT_SOURCES = {
    "synthetic": RUNS / "openvocab_namer_p3_20260708_132402_openvocab_namer.json",
    "SMD:1-1": RUNS / "openvocab_namer_1-1_p3_20260708_132402_openvocab_namer.json",
    "SMD:2-5": RUNS / "openvocab_namer_2-5_p3_20260708_132402_openvocab_namer.json",
}
DEFAULT_OUTPUT = ROOT / "docs" / "strict_naming_level_audit_2026-07-09"
METRICS = ("rule", "llm_correct", "llm_newrate", "llm_misknown")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(path: Path, background: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = payload["per_seed"]
    concepts = list(seeds[0])
    per_concept: dict[str, Any] = {}
    for concept in concepts:
        per_concept[concept] = {}
        for metric in METRICS:
            values = [float(row[concept][metric]) for row in seeds]
            per_concept[concept][metric] = {
                "mean": mean(values),
                "std_across_seeds": stdev(values) if len(values) > 1 else 0.0,
            }
    macro: dict[str, Any] = {}
    for metric in METRICS:
        seed_macros = [mean(float(row[c][metric]) for c in concepts) for row in seeds]
        macro[metric] = {
            "mean": mean(seed_macros),
            "std_across_seeds": stdev(seed_macros) if len(seed_macros) > 1 else 0.0,
        }
    return {
        "background": background,
        "source": str(path),
        "source_sha256": sha256(path),
        "n_seeds": len(seeds),
        "samples_per_concept_per_seed": 40,
        "total_calls_expected": len(seeds) * len(concepts) * 40,
        "concepts": concepts,
        "per_concept": per_concept,
        "macro": macro,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_report(results: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Strict Naming-Level Audit",
        "",
        "## Task Definitions",
        "",
        "- **Level 1, unseen-label classification:** the held-out label or an equivalent candidate definition is supplied; output is selected from a complete ontology.",
        "- **Level 2, description-grounded naming:** the novel label token is hidden, but a complete description library covers the test mechanisms.",
        "- **Level 3, free-form semantic induction:** neither novel labels nor a complete novel ontology is supplied; the model may generate a new phrase.",
        "",
        "The archived P3 run audited here is structurally a Level 3 pilot: its prompt listed only `spike`, `level_shift`, and `oscillation` as known concepts, exposed generic statistic meanings, and requested `NEW:<short name>` for unmatched evidence. Correctness was computed with type-specific keyword lists.",
        "",
        "## Archived Live-LLM Results",
        "",
        "| Background | Type | Semantic correctness | New-name rate | Misclassified as known |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        for concept in result["concepts"]:
            row = result["per_concept"][concept]
            lines.append(
                f"| {result['background']} | `{concept}` | {pct(row['llm_correct']['mean'])} | "
                f"{pct(row['llm_newrate']['mean'])} | {pct(row['llm_misknown']['mean'])} |"
            )
        macro = result["macro"]
        lines.append(
            f"| **{result['background']}** | **macro** | **{pct(macro['llm_correct']['mean'])}** | "
            f"**{pct(macro['llm_newrate']['mean'])}** | **{pct(macro['llm_misknown']['mean'])}** |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The synthetic macro semantic correctness is materially above zero, but it is not robust to real background texture. The SMD macro results are much lower even though the model still emits a new phrase at a high rate. Generation rate is therefore not a proxy for mechanism correctness.",
            "",
            "This result cannot be promoted to a primary Level 3 claim because the archived JSON contains only aggregate rates. It does not preserve each generated phrase, prompt/response pair, or evaluator decision. Human blinding, inter-rater agreement, embedding matching, and rubric-level parent/composite/abstention scores cannot be reconstructed.",
            "",
            "## Required Rerun",
            "",
            "A submission-quality Level 3 rerun must persist every raw output and prompt, freeze the model/version/sampling parameters, use two independent blinded raters, report agreement, and score exact mechanism, correct parent, supported composition, reasonable abstention, and incorrect mechanism separately.",
            "",
            "## Sources",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result['background']}`: `{result['source']}` (SHA256 `{result['source_sha256']}`).")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    missing = [path for path in DEFAULT_SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing archived inputs: {missing}")
    results = [aggregate(path, background) for background, path in DEFAULT_SOURCES.items()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "strict_naming_level_audit.json"
    output_md = args.output_dir / "strict_naming_level_audit.md"
    payload = {
        "experiment": "strict_naming_level_audit",
        "new_api_calls": 0,
        "evaluation_level": 3,
        "raw_generations_available": False,
        "human_rerating_possible": False,
        "results": results,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_report(results, output_md)
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
