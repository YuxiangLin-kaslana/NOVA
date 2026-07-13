#!/usr/bin/env python3
"""Small Level-3 naming pilot with a local model and raw-output retention.

Novel labels and their definitions are absent from the prompt. The model sees
only three known concepts plus generic evidence descriptions, and must invent a
short mechanism name or abstain. This runner is an evaluation-pipeline pilot,
not a substitute for a larger frontier-model and human-rating experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import sigla_exp.ovbench as CB  # noqa: E402


DEFAULT_MODEL = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_OUTPUT = ROOT / "docs" / "local_freeform_naming_pilot_2026-07-09"
KNOWN = ("spike", "level_shift", "oscillation")
NOVEL = ("correlation_break", "trend", "variance_burst")
KEYWORDS = {
    "correlation_break": ("correlat", "decoupl", "desync", "independ", "cross-channel", "cross channel", "synchron"),
    # Avoid bare "linear": phrases such as "non-linear volatility" are not trends.
    "trend": ("trend", "drift", "ramp", "slope", "gradual", "increas", "decreas", "monoton"),
    "variance_burst": ("varian", "volatil", "burst", "fluctuat", "unstable", "erratic", "dispers", "noisy", "noise"),
}
GENERIC_EVIDENCE = {
    "kurtosis": "prominence of isolated extreme outliers",
    "local_step": "strength of an abrupt persistent local level jump",
    "spectral_peak": "dominance of one narrow-band high-frequency component",
    "var_localiz": "localization of a scale or volatility increase to one time region",
    "lin_r2": "strength of a gradual linear progression through the window",
    "decorr": "loss of cross-channel synchrony or dependence",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(zscores: dict[str, float]) -> list[dict[str, str]]:
    system = (
        "You audit a multivariate time-series event. Three mechanisms are already known:\n"
        "- spike: isolated extreme points\n"
        "- level_shift: an abrupt persistent level jump\n"
        "- oscillation: a narrow-band high-frequency periodic component\n\n"
        "The event may instead express a mechanism outside this known vocabulary. "
        "If so, invent a concise 1-4 word mechanism name. Do not select from an unstated list. "
        "Use null when the evidence is insufficient. Return compact JSON only with keys name and confidence."
    )
    user = {
        "evidence_semantics": GENERIC_EVIDENCE,
        "evidence_zscores": zscores,
        "output_schema": {"name": "string or null", "confidence": "0 to 1"},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    ]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"name": None, "rationale": None, "confidence": None, "parse_ok": False}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"name": None, "rationale": None, "confidence": None, "parse_ok": False}
    name = value.get("name")
    confidence = value.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "name": str(name).strip() if isinstance(name, str) and name.strip().lower() != "null" else None,
        "rationale": str(value.get("rationale")) if value.get("rationale") is not None else None,
        "confidence": confidence,
        "parse_ok": True,
    }


def automatic_match(name: str | None, truth: str) -> bool:
    if name is None:
        return False
    low = name.lower()
    return any(keyword in low for keyword in KEYWORDS[truth])


def build_examples(n_per_type: int, seed: int, normal_stats_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng_stats = np.random.default_rng(seed)
    mu, sd = CB.normal_stats(rng_stats, n=normal_stats_n)
    examples: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(NOVEL):
        rng = np.random.default_rng(seed + 1000 * (concept_index + 1))
        for sample_index in range(n_per_type):
            window = CB.make_window(concept, rng)
            evidence = CB.evidence(window)
            zscores = {
                key: round((evidence[key] - mu[key]) / (sd[key] + 1e-9), 2)
                for key in CB.STATS
            }
            messages = prompt_for(zscores)
            examples.append(
                {
                    "example_id": f"{concept}:{sample_index:03d}",
                    "truth_for_evaluator_only": concept,
                    "zscores": zscores,
                    "messages": messages,
                }
            )
    return examples, {"mu": mu, "sd": sd, "normal_stats_n": normal_stats_n, "seed": seed}


def generate(
    examples: list[dict[str, Any]],
    model_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        rendered = [render_prompt(tokenizer, item["messages"]) for item in batch]
        tokens = tokenizer(rendered, return_tensors="pt", padding=True)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_lengths = tokens["attention_mask"].sum(dim=1).tolist()
        for item, sequence, prompt_length in zip(batch, generated, prompt_lengths):
            # With left padding, generated continuation begins at the padded input width.
            continuation = sequence[tokens["input_ids"].shape[1] :]
            raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            parsed = parse_json(raw)
            record = {
                **item,
                "raw_output": raw,
                "parsed": parsed,
                "automatic_keyword_match": automatic_match(parsed["name"], item["truth_for_evaluator_only"]),
                "human_rating": {
                    "rater_1": None,
                    "rater_2": None,
                    "categories": ["exact_mechanism", "correct_parent", "supported_composition", "reasonable_abstention", "incorrect"],
                },
                "prompt_tokens": int(prompt_length),
                "generated_tokens": int(len(continuation)),
            }
            outputs.append(record)
    return outputs, {
        "model_config": model.config.to_dict(),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for concept in NOVEL:
        rows = [row for row in records if row["truth_for_evaluator_only"] == concept]
        result[concept] = {
            "n": len(rows),
            "parse_rate": float(np.mean([row["parsed"]["parse_ok"] for row in rows])),
            "non_abstain_rate": float(np.mean([row["parsed"]["name"] is not None for row in rows])),
            "automatic_keyword_match": float(np.mean([row["automatic_keyword_match"] for row in rows])),
        }
    result["macro"] = {
        key: float(np.mean([result[concept][key] for concept in NOVEL]))
        for key in ("parse_rate", "non_abstain_rate", "automatic_keyword_match")
    }
    return result


def build_report(payload: dict[str, Any], output: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Local Free-Form Naming Pilot",
        "",
        "## Scope",
        "",
        "This is a small Level-3 evaluation-pipeline pilot. Novel names, candidate labels, and novel definitions are absent from the prompt. Every prompt, raw generation, parse result, and blank two-rater rubric is retained in the JSON artifact.",
        "",
        f"The local model is `{payload['model']['path']}`. It is a 0.6B-parameter model and is not comparable to the archived frontier-model run. Results measure this pipeline/model combination only.",
        "",
        "## Results",
        "",
        "| Hidden mechanism | N | JSON parse | Non-abstain | Automatic keyword match |",
        "|---|---:|---:|---:|---:|",
    ]
    for concept in NOVEL:
        row = summary[concept]
        lines.append(
            f"| `{concept}` | {row['n']} | {row['parse_rate']:.1%} | {row['non_abstain_rate']:.1%} | {row['automatic_keyword_match']:.1%} |"
        )
    macro = summary["macro"]
    lines.append(
        f"| **macro** | **{sum(summary[c]['n'] for c in NOVEL)}** | **{macro['parse_rate']:.1%}** | **{macro['non_abstain_rate']:.1%}** | **{macro['automatic_keyword_match']:.1%}** |"
    )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Keyword matching is a secondary automatic diagnostic, not the primary Level-3 score.",
            "- Human fields remain intentionally blank; no agreement statistic is reported until two independent raters complete the rubric.",
            "- Generic evidence descriptions expose what each statistic measures, but not the hidden novel taxonomy or its labels.",
            "- This controlled benchmark remains feature-aligned; feature-removal and undesigned-mechanism experiments are reported separately.",
            "",
            f"Raw artifact: `{payload['artifacts']['json']}`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-per-type", type=int, default=8)
    parser.add_argument("--normal-stats-n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    torch.manual_seed(args.seed)
    started = time.time()
    examples, calibration = build_examples(args.n_per_type, args.seed, args.normal_stats_n)
    records, model_metadata = generate(examples, args.model, args.batch_size, args.max_new_tokens)
    elapsed = time.time() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "local_freeform_naming_pilot.json"
    report_path = args.output_dir / "local_freeform_naming_pilot.md"
    payload = {
        "experiment": "local_freeform_naming_pilot",
        "evaluation_level": 3,
        "novel_labels_in_prompt": False,
        "complete_novel_ontology_in_prompt": False,
        "candidate_list_in_prompt": False,
        "raw_outputs_retained": True,
        "human_ratings_complete": False,
        "model": {
            "path": str(args.model),
            "model_weight_sha256": file_sha256(args.model / "model.safetensors"),
            **model_metadata,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "calibration": calibration,
        "summary": summarize(records),
        "records": records,
        "artifacts": {"json": str(result_path), "report": str(report_path)},
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_report(payload, report_path)
    print(result_path)
    print(report_path)


if __name__ == "__main__":
    main()
