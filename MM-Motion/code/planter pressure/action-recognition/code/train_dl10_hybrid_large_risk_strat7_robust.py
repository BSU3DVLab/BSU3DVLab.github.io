#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL10 Hybrid Large — evaluation under risk-stratified subject holdout.

Outer protocol (aligned with risk experiments):
  - Exclude subject22–28 and subject32.
  - Each run: sample exactly 7 test subjects with counts (高风险×2, 中风险×3, 低风险×2).
  - Train only on remaining subjects; z-score from TRAIN file paths only.
  - Sliding windows L=160 (train random-offset via split_name=train / test center-equivalent windows).

For each run: write run_*/train.npz & test.npz, train HybridDL10Large with best-by-test-trial selection,
then reload best checkpoint and record window + trial metrics.

Dependencies: prepare_dl10_holdout32_step3 (build_split_arrays, fit_zscore_stats),
train_dl10_hybrid_large_holdout (train, evaluate, ...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import prepare_dl10_holdout32_step3 as prep
import train_dl10_hybrid_large_holdout as H
from run_hybrid_multitask_ordinal_robust30 import sample_balanced_test, summarize


def build_pool_by_risk(df: pd.DataFrame) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {"高风险": [], "中风险": [], "低风险": []}
    for sid, rl in df.drop_duplicates("subject_id")[["subject_id", "risk_level"]].itertuples(index=False):
        sid, rl = str(sid), str(rl)
        if rl in pool:
            pool[rl].append(sid)
    return pool


def load_weights_optional(path: str) -> Dict[str, float] | None:
    p = str(path).strip()
    if not p:
        return None
    return prep.load_subject_weights_kg(p)


def fit_mean_std_weighted(
    train_df: pd.DataFrame,
    weights: Dict[str, float] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    paths = train_df["file_path"].astype(str).tolist()
    if weights is None:
        return prep.fit_zscore_stats(paths)
    total_frames = 0
    sum_x = np.zeros(96, dtype=np.float64)
    sum_x2 = np.zeros(96, dtype=np.float64)
    for row in train_df.itertuples(index=False):
        x = prep.read_sensor_matrix(str(row.file_path))
        if x.shape[0] == 0:
            continue
        sid = str(row.subject_id)
        wk = weights.get(sid)
        if wk is None:
            idx = prep._parse_subject_index(sid)
            if idx is not None:
                wk = weights.get(f"subject{idx:02d}")
        if wk is None:
            raise KeyError(f"Missing weight for subject_id={sid}")
        x = prep.normalize_by_weight(x, wk)
        sum_x += x.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_frames += x.shape[0]
    if total_frames <= 0:
        raise RuntimeError("No frames for z-score with weights.")
    mean = (sum_x / total_frames).astype(np.float32)
    var = (sum_x2 / total_frames - np.square(mean)).astype(np.float32)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="CSV with file_path,subject_id,pose_id,risk_level")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=51000)
    ap.add_argument("--length", type=int, default=160)
    ap.add_argument("--window-stride", type=int, default=80)
    ap.add_argument("--pad-mode", type=str, default="zero", choices=["zero", "end_repeat"])
    ap.add_argument(
        "--weight-csv",
        type=str,
        default="",
        help="Optional CAIT weight CSV for normalize_by_weight; empty => raw g + z-score only.",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=7e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--n-classes", type=int, default=10)
    args = ap.parse_args()

    excluded: Set[str] = {f"subject{i:02d}" for i in range(22, 29)} | {"subject32"}
    df = pd.read_csv(Path(args.manifest), encoding="utf-8-sig")
    df = df[~df["subject_id"].astype(str).isin(excluded)].reset_index(drop=True)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(int) == 1].reset_index(drop=True)
    need = {"file_path", "subject_id", "pose_id", "risk_level"}
    if not need.issubset(df.columns):
        raise ValueError(f"manifest must contain columns: {need}")

    pool_by_risk = build_pool_by_risk(df)
    all_subjects = sorted(df["subject_id"].astype(str).unique().tolist())
    weights = load_weights_optional(args.weight_csv)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    acc_w = []
    acc_t = []
    f1_w = []
    f1_t = []

    for i in range(int(args.n_runs)):
        seed = int(args.seed0) + i
        rng = np.random.default_rng(seed)
        test_sub = sample_balanced_test(rng, pool_by_risk)
        test_set = set(test_sub)
        train_sub = sorted(set(all_subjects) - test_set)
        train_set = set(train_sub)

        train_df = df[df["subject_id"].astype(str).isin(train_set)].reset_index(drop=True)
        test_df = df[df["subject_id"].astype(str).isin(test_set)].reset_index(drop=True)

        mean, std = fit_mean_std_weighted(train_df, weights)
        train_data = prep.build_split_arrays(
            split_df=train_df,
            mean=mean,
            std=std,
            length=int(args.length),
            split_name="train",
            seed=seed,
            pad_mode=str(args.pad_mode),
            use_windows=True,
            window_stride=int(args.window_stride),
            subject_weights_kg=weights,
        )
        test_data = prep.build_split_arrays(
            split_df=test_df,
            mean=mean,
            std=std,
            length=int(args.length),
            split_name="test",
            seed=seed,
            pad_mode=str(args.pad_mode),
            use_windows=True,
            window_stride=int(args.window_stride),
            subject_weights_kg=weights,
        )

        run_dir = out_root / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "train_mean.npy", mean)
        np.save(run_dir / "train_std.npy", std)
        np.savez_compressed(run_dir / "train.npz", **train_data)
        np.savez_compressed(run_dir / "test.npz", **test_data)

        run_name = f"strat7_r{i}"
        train_args = SimpleNamespace(
            train_npz=str(run_dir / "train.npz"),
            test_npz=str(run_dir / "test.npz"),
            out_dir=str(run_dir),
            run_name=run_name,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            dropout=float(args.dropout),
            gamma=float(args.gamma),
            seed=seed,
            n_classes=int(args.n_classes),
        )
        H.set_seed(seed)
        H.train(train_args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = run_dir / f"hybrid_large_holdout_{run_name}_best_by_test.pt"
        if not ckpt.exists():
            ckpt = run_dir / f"hybrid_large_holdout_{run_name}_last.pt"
        model = H.HybridDL10Large(n_classes=int(args.n_classes), dropout=float(args.dropout)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        test_ds = H.NpzSequenceDataset(run_dir / "test.npz")
        test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        wm = H.evaluate(model, test_loader, device)
        tm = H.evaluate_trial_aggregated(model, test_loader, device)

        rec = {
            "run": i,
            "seed": seed,
            "test_subjects": test_sub,
            "n_train_windows": int(train_data["X"].shape[0]),
            "n_test_windows": int(test_data["X"].shape[0]),
            "window": wm,
            "trial": tm if tm is not None else {},
        }
        records.append(rec)
        acc_w.append(wm["accuracy"])
        f1_w.append(wm["macro_f1"])
        if tm is not None:
            acc_t.append(tm["accuracy"])
            f1_t.append(tm["macro_f1"])

    acc_w = np.asarray(acc_w, dtype=np.float64)
    f1_w = np.asarray(f1_w, dtype=np.float64)
    summary = {
        "n_runs": int(args.n_runs),
        "rule": "test = 7 subjects (2高/3中/2低), exclude subject22-28,32; z-score on train files only",
        "model": "HybridDL10Large (dual TCN + Transformer + stat branch + head)",
        "data": {
            "length": int(args.length),
            "window_stride": int(args.window_stride),
            "weight_csv": str(args.weight_csv) or None,
        },
        "train": {"epochs": int(args.epochs), "batch_size": int(args.batch_size), "lr": float(args.lr)},
        "window_test": {"accuracy": summarize(acc_w), "macro_f1": summarize(f1_w)},
    }
    if f1_t:
        summary["trial_test"] = {
            "accuracy": summarize(np.asarray(acc_t, dtype=np.float64)),
            "macro_f1": summarize(np.asarray(f1_t, dtype=np.float64)),
        }

    (out_root / "runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "report.txt").write_text(
        "DL10 Hybrid Large — risk-stratified 7-subject test\n" + json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out_root / "report.txt")


if __name__ == "__main__":
    main()
