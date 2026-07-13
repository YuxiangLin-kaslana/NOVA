#!/usr/bin/env python3
"""Prepare public NOVA figure-set datasets into the local project layout.

This script is intentionally conservative: it prepares datasets that are already
available locally or can be copied from a fetched public source, and records
datasets that require credentials or applications as unavailable instead of
fabricating placeholders.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = DATA / "nova_datasets_manifest.json"
PSM_SRC = Path("/tmp/ransyncoders-data/data")
NASA_RAW = DATA / "NASA_SMAP_MSL" / "raw" / "extracted"


def _fill_missing_features(arr: np.ndarray) -> tuple[np.ndarray, int]:
    missing = int(np.isnan(arr).sum())
    if missing == 0:
        return arr.astype(np.float32), 0

    out = arr.astype(np.float64, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        good = np.flatnonzero(~np.isnan(col))
        if len(good) == 0:
            out[:, j] = 0.0
            continue
        first = good[0]
        if first > 0:
            col[:first] = col[first]
        for k in range(first + 1, len(col)):
            if np.isnan(col[k]):
                col[k] = col[k - 1]
        if np.isnan(col).any():
            med = float(np.nanmedian(col))
            col[np.isnan(col)] = med if np.isfinite(med) else 0.0
        out[:, j] = col
    return out.astype(np.float32), missing


def read_csv_matrix(path: Path, *, label: bool = False) -> tuple[np.ndarray, int]:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if label:
            if header != ["timestamp_(min)", "label"]:
                raise ValueError(f"unexpected PSM label header in {path}: {header}")
            for row in reader:
                rows.append([float(row[1])])
            return np.asarray(rows, dtype=np.float32).reshape(-1), 0

        if len(header) < 2 or header[0] != "timestamp_(min)":
            raise ValueError(f"unexpected PSM feature header in {path}: {header[:5]}")
        for row in reader:
            rows.append([float(x) if x != "" else np.nan for x in row[1:]])
    return _fill_missing_features(np.asarray(rows, dtype=np.float64))


def dump_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def summarize_smd() -> dict:
    pre = DATA / "ServerMachineDataset" / "preprocessed"
    train = sorted(pre.glob("machine-*_train.pkl"))
    test = sorted(pre.glob("machine-*_test.pkl"))
    label = sorted(pre.glob("machine-*_test_label.pkl"))
    entities = sorted(p.name.replace("_train.pkl", "") for p in train)
    sample_shape = None
    if train:
        sample = pickle.load(train[0].open("rb"))
        sample_shape = list(np.asarray(sample).shape)
    complete = len(train) == len(test) == len(label) == 28
    return {
        "status": "available" if complete else "partial",
        "role": "main real multivariate background and native binary labels",
        "path": str(DATA / "ServerMachineDataset"),
        "entities": len(entities),
        "expected_entities": 28,
        "train_files": len(train),
        "test_files": len(test),
        "test_label_files": len(label),
        "sample_train_shape": sample_shape,
    }


def prepare_psm() -> dict:
    out = DATA / "PSM"
    raw = out / "raw"
    pre = out / "preprocessed"
    required = ["train.csv", "test.csv", "test_label.csv", "LICENSE"]
    source = PSM_SRC if PSM_SRC.exists() else raw
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        return {
            "status": "missing_source",
            "role": "public eBay server metrics extension dataset",
            "path": str(out),
            "missing_source_files": missing,
            "source_expected": str(PSM_SRC),
            "raw_expected": str(raw),
        }

    raw.mkdir(parents=True, exist_ok=True)
    for name in required:
        if source.resolve() != raw.resolve():
            shutil.copy2(source / name, raw / name)

    train, train_missing = read_csv_matrix(raw / "train.csv")
    test, test_missing = read_csv_matrix(raw / "test.csv")
    label, _ = read_csv_matrix(raw / "test_label.csv", label=True)
    label = label.astype(np.int64)
    if len(test) != len(label):
        raise ValueError(f"PSM test/label length mismatch: {len(test)} vs {len(label)}")

    dump_pickle(train, pre / "psm_train.pkl")
    dump_pickle(test, pre / "psm_test.pkl")
    dump_pickle(label, pre / "psm_test_label.pkl")

    summary = {
        "dataset": "PSM",
        "source": "https://github.com/eBay/RANSynCoders/tree/main/data",
        "license_file": str(raw / "LICENSE"),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "test_label_shape": list(label.shape),
        "anomaly_points": int(label.sum()),
        "anomaly_ratio": float(label.mean()),
        "missing_values_filled": {"train": train_missing, "test": test_missing},
        "feature_columns": [f"feature_{i}" for i in range(train.shape[1])],
        "format": {
            "raw": ["raw/train.csv", "raw/test.csv", "raw/test_label.csv"],
            "preprocessed": [
                "preprocessed/psm_train.pkl",
                "preprocessed/psm_test.pkl",
                "preprocessed/psm_test_label.pkl",
            ],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return {
        "status": "available",
        "role": "public eBay server metrics extension dataset; recommended first extension in novafigureset.pdf",
        "path": str(out),
        **summary,
    }


def prepare_nasa_spacecraft(spacecraft: str) -> dict:
    """Prepare Telemanom/Kaggle SMAP or MSL channels into generic realbench layout."""
    sc = spacecraft.upper()
    lab = NASA_RAW / "labeled_anomalies.csv"
    arr_root = NASA_RAW / "data" / "data"
    if not lab.exists() or not (arr_root / "train").exists() or not (arr_root / "test").exists():
        return {
            "status": "not_downloaded_public_credentials_needed",
            "role": f"NASA/JPL {sc} spacecraft telemetry extension dataset",
            "note": "Kaggle/Telemanom raw files are not present. Use Kaggle API token to download patrickfleith/nasa-anomaly-detection-dataset-smap-msl.",
            "source": "https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        }

    out = DATA / sc
    pre = out / "preprocessed"
    pre.mkdir(parents=True, exist_ok=True)
    rows = []
    with lab.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["spacecraft"].upper() == sc:
                rows.append(row)

    entities = []
    total_test = 0
    total_anom = 0
    dims: set[int] = set()
    for row in rows:
        cid = row["chan_id"]
        train = np.load(arr_root / "train" / f"{cid}.npy").astype(np.float32)
        test = np.load(arr_root / "test" / f"{cid}.npy").astype(np.float32)
        seqs = json.loads(row["anomaly_sequences"])
        label = np.zeros(len(test), dtype=np.int64)
        for start, end in seqs:
            s = max(0, int(start))
            e = min(len(label) - 1, int(end))
            if e >= s:
                label[s : e + 1] = 1
        dump_pickle(train, pre / f"{cid}_train.pkl")
        dump_pickle(test, pre / f"{cid}_test.pkl")
        dump_pickle(label, pre / f"{cid}_test_label.pkl")
        dims.add(int(train.shape[1]))
        total_test += int(len(test))
        total_anom += int(label.sum())
        entities.append(
            {
                "entity": cid,
                "train_shape": list(train.shape),
                "test_shape": list(test.shape),
                "anomaly_sequences": seqs,
                "anomaly_points": int(label.sum()),
            }
        )

    summary = {
        "dataset": sc,
        "source": "https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "raw_root": str(NASA_RAW),
        "preprocessed": str(pre),
        "entity_count": len(entities),
        "feature_dims": sorted(dims),
        "total_test_points": total_test,
        "total_anomaly_points": total_anom,
        "anomaly_ratio": float(total_anom / max(1, total_test)),
        "entities": entities,
        "format": "generic realbench layout: preprocessed/<entity>_{train,test,test_label}.pkl",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return {
        "status": "available",
        "role": f"NASA/JPL {sc} telemetry extension dataset",
        "path": str(out),
        "entity_count": len(entities),
        "feature_dims": sorted(dims),
        "total_test_points": total_test,
        "total_anomaly_points": total_anom,
        "anomaly_ratio": summary["anomaly_ratio"],
        "summary": str(out / "summary.json"),
        "source": summary["source"],
    }


def index_ucr() -> dict:
    root = DATA / "UCR"
    full = (
        root
        / "raw"
        / "extracted"
        / "AnomalyDatasets_2021"
        / "UCR_TimeSeriesAnomalyDatasets2021"
        / "FilesAreInHere"
        / "UCR_Anomaly_FullData"
    )
    zip_path = root / "raw" / "UCR_TimeSeriesAnomalyDatasets2021.zip"
    if not full.exists():
        return {
            "status": "optional_not_downloaded",
            "role": "optional native-label binary detection bridge check",
            "path": str(root),
            "note": "Optional in novafigureset.pdf; raw archive not present locally.",
            "source": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip",
        }

    rows = []
    pat = re.compile(r"^(?P<id>\d+)_UCR_Anomaly_(?P<name>.+)_(?P<train_end>\d+)_(?P<anom_start>\d+)_(?P<anom_end>\d+)\.txt$")
    for p in sorted(full.glob("*.txt")):
        m = pat.match(p.name)
        item = {"file": str(p), "basename": p.name, "parse_ok": bool(m)}
        if m:
            item.update(
                {
                    "id": int(m.group("id")),
                    "series_name": m.group("name"),
                    "train_end": int(m.group("train_end")),
                    "anomaly_start": int(m.group("anom_start")),
                    "anomaly_end": int(m.group("anom_end")),
                }
            )
        rows.append(item)

    index = {
        "dataset": "UCR Anomaly Archive 2021",
        "source": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip",
        "archive": str(zip_path),
        "full_data_dir": str(full),
        "series_count": len(rows),
        "parsed_count": sum(1 for r in rows if r["parse_ok"]),
        "format": "univariate txt; filename encodes train_end, anomaly_start, anomaly_end",
        "series": rows,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return {
        "status": "available_optional",
        "role": "optional native-label binary detection bridge check",
        "path": str(root),
        "series_count": index["series_count"],
        "parsed_count": index["parsed_count"],
        "index": str(root / "index.json"),
        "source": index["source"],
    }


def time_mmd_status() -> dict:
    root = DATA / "data" / "Time-MMD"
    domains = []
    for num in sorted((root / "numerical").glob("*/*.csv")):
        dom = num.parent.name
        text = root / "textual" / dom / f"{dom}_report.csv"
        domains.append({"domain": dom, "numerical": str(num), "textual": str(text), "textual_exists": text.exists()})
    return {
        "status": "available" if domains else "missing",
        "role": "multimodal numerical + aligned reports for A7/cost-aware inspection",
        "path": str(root),
        "domains": domains,
        "domain_count": len(domains),
    }


def placeholder_status(name: str, status: str, role: str, note: str, source: str | None = None) -> dict:
    out = {"status": status, "role": role, "note": note}
    if source:
        out["source"] = source
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_by": "code/sota_compare/prepare_nova_datasets.py",
        "figure_set_pdf": str(ROOT / "docs" / "novafigureset.pdf"),
        "required_by_pdf": {
            "real_multivariate": ["Synthetic/ovbench", "SMD", "MSL", "SMAP", "PSM", "SWaT", "WADI"],
            "optional_native_binary": ["UCR Anomaly Archive"],
            "multimodal": ["Time-MMD", "MIMIC-IV"],
        },
        "datasets": {
            "Synthetic/ovbench": {
                "status": "code_available",
                "role": "controlled synthetic mechanism benchmark",
                "path": str(ROOT / "code" / "sigla_exp" / "ovbench.py"),
                "concepts": [
                    "spike",
                    "level_shift",
                    "oscillation",
                    "variance_burst",
                    "trend",
                    "correlation_break",
                ],
            },
            "SMD": summarize_smd(),
            "PSM": prepare_psm(),
            "MSL": prepare_nasa_spacecraft("MSL"),
            "SMAP": prepare_nasa_spacecraft("SMAP"),
            "SWaT": placeholder_status(
                "SWaT",
                "requires_application_not_filled",
                "industrial control cross-domain dataset",
                "Requires iTrust/SUTD access approval; left as README placeholder per user instruction.",
                str(DATA / "SWaT" / "README.md"),
            ),
            "WADI": placeholder_status(
                "WADI",
                "requires_application_not_filled",
                "high-dimensional industrial control dataset",
                "Requires iTrust/SUTD access approval; left as README placeholder per user instruction.",
                str(DATA / "WADI" / "README.md"),
            ),
            "UCR": index_ucr(),
            "Time-MMD": time_mmd_status(),
            "MIMIC-IV": placeholder_status(
                "MIMIC-IV",
                "requires_credentials_not_filled",
                "ICU vitals + notes multimodal positive case",
                "Requires PhysioNet/CITI/DUA credentials; code is staged but data is not copied.",
                str(ROOT / "code" / "mimic"),
            ),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST}")
    print(json.dumps(manifest["datasets"]["PSM"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
