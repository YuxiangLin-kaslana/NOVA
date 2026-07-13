#!/usr/bin/env python3
"""V2 taxonomy expansion stress test.

Paper question:
  Do the new V2 anomaly types support a stronger objective than binary
  detection?  We test type-aware open-vocabulary adaptation on the three
  taxonomy additions that are not well represented by the older six-type
  open-vocabulary benchmark: seasonal_break, contextual_deviation, shapelet.

Protocol:
  - Pretrain a closed detector on normal + three known V2 primitives:
    spike_burst, level_shift, correlation_break.
  - Stream introduces seasonal_break, contextual_deviation, and shapelet in
    staggered phases, without labels.
  - Frozen baseline can only output the closed vocabulary.
  - Bootstrap uses label-free evidence to name/grow a new type and replays
    balanced pseudo-labels.

Default namer is deterministic evidence-to-concept mapping for reproducibility.
Set V2T_NAMER=gpt to call gpt-4o-mini when OPENAI_API_KEY is available.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigla_exp.model import CNNConceptDetector  # noqa: E402
import sigla_exp.v2bench as VB  # noqa: E402


KNOWN = ["spike_burst", "level_shift", "correlation_break"]
NOVELS = ["seasonal_break", "contextual_deviation", "shapelet"]
BASE_VOCAB = [VB.NORMAL] + KNOWN
KNOWN_STATS = {VB.STAT_OF[c] for c in KNOWN}

SMOKE = os.environ.get("V2T_SMOKE", "0") == "1"
NSEED = int(os.environ.get("V2T_NSEED", "1" if SMOKE else "5"))
NAMER = os.environ.get("V2T_NAMER", "rule").lower()
TAU = float(os.environ.get("V2T_TAU", "0.55"))
NOVEL_Z = float(os.environ.get("V2T_NOVEL_Z", "2.2"))
AUDIT = float(os.environ.get("V2T_AUDIT", "0.04"))
RETRAIN_EVERY = int(os.environ.get("V2T_RETRAIN_EVERY", "12"))

N_PRETRAIN = int(os.environ.get("V2T_N_PRETRAIN", "200" if SMOKE else "2600"))
PRETRAIN_EPOCHS = int(os.environ.get("V2T_PRETRAIN_EPOCHS", "2" if SMOKE else "24"))
N_WARM = int(os.environ.get("V2T_N_WARM", "28" if SMOKE else "140"))
SEG = int(os.environ.get("V2T_SEG", "25" if SMOKE else "180"))
REPLAY_PER_CLASS = int(os.environ.get("V2T_REPLAY_PER_CLASS", "10" if SMOKE else "36"))
REFIT_EPOCHS = int(os.environ.get("V2T_REFIT_EPOCHS", "1" if SMOKE else "2"))
DET_MIN_CONF = float(os.environ.get("V2T_DET_MIN_CONF", "0.45"))
NORMAL_MARGIN = float(os.environ.get("V2T_NORMAL_MARGIN", "0.05"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(int(os.environ.get("V2T_TORCH_THREADS", "1" if SMOKE else "2")))


def make_detector(n_out: int) -> CNNConceptDetector:
    channels = (16, 32) if SMOKE else (64, 128)
    return CNNConceptDetector(VB.WIN, VB.NVARS, n_concepts=n_out, channels=channels, kernel_size=7).to(device)


@torch.no_grad()
def proba(det: CNNConceptDetector, windows: list[np.ndarray]) -> np.ndarray:
    det.eval()
    x = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    return torch.sigmoid(det(x)).detach().cpu().numpy()


def onehot(idx: int, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    y[idx] = 1.0
    return y


def pick_detector_label(vocab: list[str], probs: np.ndarray) -> tuple[str, int]:
    pred_idx = int(np.argmax(probs))
    if VB.NORMAL in vocab:
        normal_idx = vocab.index(VB.NORMAL)
        anom = [i for i, name in enumerate(vocab) if name != VB.NORMAL]
        max_anom = float(np.max(probs[anom])) if anom else 0.0
        if float(probs.max()) < DET_MIN_CONF:
            return VB.NORMAL, normal_idx
        if float(probs[normal_idx]) + NORMAL_MARGIN >= max_anom:
            return VB.NORMAL, normal_idx
    return vocab[pred_idx], pred_idx


def train_on(det: CNNConceptDetector, opt: torch.optim.Optimizer, x_list: list[np.ndarray], y_list: list[np.ndarray], epochs: int) -> None:
    det.train()
    x = torch.tensor(np.stack(x_list), dtype=torch.float32, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 96):
            idx = perm[start : start + 96]
            loss = F.binary_cross_entropy_with_logits(det(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    det.eval()


def grow_head(det: CNNConceptDetector, new_n: int) -> None:
    old = det.head[-1]
    new = nn.Linear(old.in_features, new_n).to(device)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
    det.head[-1] = new


def make_labeled(concept: str, rng: np.random.Generator) -> np.ndarray:
    return VB.make_window(None if concept == VB.NORMAL else concept, rng)


def build_pretrain(rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(N_PRETRAIN):
        concept = BASE_VOCAB[int(rng.integers(len(BASE_VOCAB)))]
        x_list.append(make_labeled(concept, rng))
        idx = BASE_VOCAB.index(concept)
        y_list.append(onehot(idx, len(BASE_VOCAB)))
        labels.append(idx)
    return x_list, y_list, labels


def build_stream(rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, str]], dict[str, int]]:
    stream: list[tuple[np.ndarray, str]] = []
    for _ in range(N_WARM):
        concept = VB.NORMAL if rng.random() < 0.28 else KNOWN[int(rng.integers(len(KNOWN)))]
        stream.append((make_labeled(concept, rng), concept))

    onsets: dict[str, int] = {}
    active_novels: list[str] = []
    for novel in NOVELS:
        active_novels.append(novel)
        onsets[novel] = len(stream)
        for _ in range(SEG):
            r = rng.random()
            if r < 0.24:
                concept = VB.NORMAL
            elif r < 0.56:
                concept = KNOWN[int(rng.integers(len(KNOWN)))]
            else:
                concept = active_novels[int(rng.integers(len(active_novels)))]
            stream.append((make_labeled(concept, rng), concept))
    return stream, onsets


def evidence_namer(ev: dict[str, float], mu: dict[str, float], sd: dict[str, float], key: str, net_ok: bool) -> str | None:
    if NAMER == "gpt" and net_ok:
        got = VB.gpt_recognize_top1(ev, key, mu, sd)
        if got != "__ERROR__":
            return got
    return VB.rule_namer(ev, mu, sd, threshold=NOVEL_Z)


def suspect_novel(ev: dict[str, float], mu: dict[str, float], sd: dict[str, float]) -> tuple[bool, str | None, dict[str, float]]:
    z = VB.z_scores(ev, mu, sd)
    dom = max(z, key=z.get)
    concept = VB.CONCEPT_OF_STAT[dom]
    return (dom not in KNOWN_STATS and z[dom] >= NOVEL_Z), concept, z


def class_metrics(preds: list[str], trues: list[str], start: int = 0) -> dict[str, float]:
    p = preds[start:]
    t = trues[start:]
    novel_mask = [true in NOVELS for true in t]
    normal_mask = [true == VB.NORMAL for true in t]
    novel_typed = [pred == true for pred, true, m in zip(p, t, novel_mask) if m]
    novel_detected = [pred != VB.NORMAL for pred, true, m in zip(p, t, novel_mask) if m]
    normal_fp = [pred != VB.NORMAL for pred, true, m in zip(p, t, normal_mask) if m]
    all_correct = [pred == true for pred, true in zip(p, t)]
    return {
        "overall_type_acc": float(np.mean(all_correct)) if all_correct else 0.0,
        "novel_typed_acc": float(np.mean(novel_typed)) if novel_typed else 0.0,
        "novel_detection_recall": float(np.mean(novel_detected)) if novel_detected else 0.0,
        "normal_false_alarm": float(np.mean(normal_fp)) if normal_fp else 0.0,
    }


def run_seed(seed: int, key: str, net_ok: bool) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print(f"[seed {seed}] calibrating evidence", flush=True)
    mu, sd = VB.normal_stats(rng, n=32 if SMOKE else 300)

    print(f"[seed {seed}] pretraining detector", flush=True)
    det = make_detector(len(BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    xpt, ypt, replay_labels = build_pretrain(rng)
    train_on(det, opt, xpt, ypt, epochs=PRETRAIN_EPOCHS)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(xpt, replay_labels))

    stream, onsets = build_stream(rng)
    trues = [true for _, true in stream]

    print(f"[seed {seed}] frozen pass", flush=True)
    frozen_pred: list[str] = []
    for x, _ in stream:
        ev = VB.evidence(x)
        if VB.rule_namer(ev, mu, sd, threshold=NOVEL_Z) is None:
            frozen_pred.append(VB.NORMAL)
            continue
        p = proba(det, [x])[0]
        pred, _ = pick_detector_label(BASE_VOCAB, p)
        frozen_pred.append(pred)

    det = make_detector(len(BASE_VOCAB))
    det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(BASE_VOCAB)
    buffer: list[tuple[np.ndarray, int]] = []
    boot_pred: list[str] = []
    call_flags: list[int] = []
    vocab_trace: list[int] = []
    pending = 0

    print(f"[seed {seed}] bootstrap pass", flush=True)
    for x, _true in stream:
        p = proba(det, [x])[0]
        mx = float(p.max())
        pred, pred_idx = pick_detector_label(vocab, p)
        ev = VB.evidence(x)
        if VB.rule_namer(ev, mu, sd, threshold=NOVEL_Z) is None:
            boot_pred.append(VB.NORMAL)
            call_flags.append(0)
            vocab_trace.append(len(vocab))
            continue
        susp, _concept, _z = suspect_novel(ev, mu, sd)
        mislabel = pred in BASE_VOCAB and susp
        audit = rng.random() < AUDIT
        called = int(mx < TAU or mislabel or audit)

        if called:
            named = evidence_namer(ev, mu, sd, key, net_ok)
            if named in VB.CONCEPTS:
                pred = named
                if named not in vocab:
                    if susp:
                        vocab.append(named)
                        grow_head(det, len(vocab))
                        opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                    else:
                        pred = vocab[pred_idx]
            if pred in vocab:
                buffer.append((x, vocab.index(pred)))
                pending += 1

        if pending >= RETRAIN_EVERY:
            per_class: dict[int, list[np.ndarray]] = {}
            for w, label_idx in replay:
                per_class.setdefault(label_idx, []).append(w)
            for w, label_idx in buffer:
                per_class.setdefault(label_idx, []).append(w)
            xb: list[np.ndarray] = []
            yb: list[np.ndarray] = []
            for label_idx, windows in per_class.items():
                draws = rng.integers(0, len(windows), size=REPLAY_PER_CLASS)
                for j in draws:
                    xb.append(windows[int(j)])
                    yb.append(onehot(label_idx, len(vocab)))
            train_on(det, opt, xb, yb, epochs=REFIT_EPOCHS)
            pending = 0

        boot_pred.append(pred)
        call_flags.append(called)
        vocab_trace.append(len(vocab))

    first_novel = min(onsets.values())
    last_novel = onsets[NOVELS[-1]]
    per_novel = {}
    for novel in NOVELS:
        idx = [i for i, true in enumerate(trues) if true == novel]
        chunks = [chunk for chunk in np.array_split(idx, 4) if len(chunk)]
        per_novel[novel] = {
            "n": len(idx),
            "frozen_typed_acc": float(np.mean([frozen_pred[i] == novel for i in idx])) if idx else 0.0,
            "bootstrap_typed_acc": float(np.mean([boot_pred[i] == novel for i in idx])) if idx else 0.0,
            "bootstrap_curve": [float(np.mean([boot_pred[i] == novel for i in chunk])) for chunk in chunks],
        }

    return {
        "frozen": class_metrics(frozen_pred, trues, start=first_novel),
        "bootstrap": class_metrics(boot_pred, trues, start=first_novel),
        "frozen_last": class_metrics(frozen_pred, trues, start=last_novel),
        "bootstrap_last": class_metrics(boot_pred, trues, start=last_novel),
        "per_novel": per_novel,
        "call_rate": float(np.mean(call_flags[first_novel:])) if len(call_flags) > first_novel else 0.0,
        "final_vocab": vocab,
        "grew_all": int(all(novel in vocab for novel in NOVELS)),
        "vocab_curve": [int(vocab_trace[chunk[-1]]) for chunk in np.array_split(np.arange(len(vocab_trace)), 10) if len(chunk)],
        "onsets": onsets,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def print_separation(sep: dict[str, object]) -> None:
    rows = sep["rows"]
    print("=== V2 evidence signature separation ===")
    print(f"{'concept':24s}{'signature':22s}{'dominant':22s}{'hit':>6s}")
    for concept in VB.CONCEPTS:
        row = rows[concept]
        print(f"{concept:24s}{row['signature']:22s}{row['dominant']:22s}{str(row['hit']):>6s}")
    print()


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    sep = VB.signature_separation(np.random.default_rng(777), n=6 if SMOKE else 80)
    print(f"device={device} smoke={SMOKE} nseed={NSEED} namer={NAMER} net_ok={net_ok}")
    print(f"known={KNOWN}  novels={NOVELS}  base_vocab={BASE_VOCAB}")
    print(f"TAU={TAU} NOVEL_Z={NOVEL_Z} AUDIT={AUDIT}\n")
    print(
        f"sizes: pretrain={N_PRETRAIN} epochs={PRETRAIN_EPOCHS} warm={N_WARM} "
        f"seg={SEG} replay_per_class={REPLAY_PER_CLASS} det_min_conf={DET_MIN_CONF}\n"
    )
    print_separation(sep)

    res = []
    for seed in range(NSEED):
        out = run_seed(seed, key, net_ok)
        res.append(out)
        print(
            f"[seed {seed}] novel typed frozen={out['frozen']['novel_typed_acc']:.0%} "
            f"boot={out['bootstrap']['novel_typed_acc']:.0%} | "
            f"last overall frozen={out['frozen_last']['overall_type_acc']:.0%} "
            f"boot={out['bootstrap_last']['overall_type_acc']:.0%} | "
            f"FAR boot={out['bootstrap']['normal_false_alarm']:.0%} "
            f"calls={out['call_rate']:.0%} vocab={out['final_vocab']}"
        )

    print("\n" + "=" * 86)
    for metric in ["overall_type_acc", "novel_typed_acc", "novel_detection_recall", "normal_false_alarm"]:
        fm, fs = mean_std([r["frozen"][metric] for r in res])
        bm, bs = mean_std([r["bootstrap"][metric] for r in res])
        print(f"{metric:24s}: frozen {fm:.2f}±{fs:.2f}   bootstrap {bm:.2f}±{bs:.2f}")
    call_m, call_s = mean_std([r["call_rate"] for r in res])
    print(f"{'namer_call_rate':24s}: bootstrap {call_m:.2f}±{call_s:.2f}")
    print(f"{'vocab_grew_all':24s}: {np.mean([r['grew_all'] for r in res]):.0%}")

    per_novel = {}
    print("\nPer-new-type typed accuracy:")
    print(f"{'novel':24s}{'frozen':>12s}{'bootstrap':>16s}{'curve':>28s}")
    for novel in NOVELS:
        fm, _ = mean_std([r["per_novel"][novel]["frozen_typed_acc"] for r in res])
        bm, bs = mean_std([r["per_novel"][novel]["bootstrap_typed_acc"] for r in res])
        max_len = max(len(r["per_novel"][novel]["bootstrap_curve"]) for r in res)
        curves = []
        for i in range(max_len):
            vals = [r["per_novel"][novel]["bootstrap_curve"][i] for r in res if i < len(r["per_novel"][novel]["bootstrap_curve"])]
            curves.append(float(np.mean(vals)))
        per_novel[novel] = {
            "frozen_mean": fm,
            "bootstrap_mean": bm,
            "bootstrap_std": bs,
            "curve_mean": curves,
        }
        print(f"{novel:24s}{fm:12.0%}{bm:12.0%}±{bs:<3.0%}{str([round(v, 2) for v in curves]):>28s}")
    print("=" * 86)

    out_path = ROOT / "runs" / "v2_taxonomy_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "known": KNOWN,
        "novels": NOVELS,
        "nseed": NSEED,
        "smoke": SMOKE,
        "namer": NAMER,
        "separation": sep,
        "aggregate": {
            metric: {
                "frozen": dict(zip(["mean", "std"], mean_std([r["frozen"][metric] for r in res]))),
                "bootstrap": dict(zip(["mean", "std"], mean_std([r["bootstrap"][metric] for r in res]))),
            }
            for metric in ["overall_type_acc", "novel_typed_acc", "novel_detection_recall", "normal_false_alarm"]
        },
        "per_novel": per_novel,
        "per_seed": res,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
