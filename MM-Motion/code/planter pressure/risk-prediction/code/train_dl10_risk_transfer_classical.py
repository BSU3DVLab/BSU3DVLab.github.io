#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Risk classification via transfer features from pretrained DL10 Hybrid.

Pipeline:
1) Build fixed-length windows (center crop) and z-score (fit on train subjects only).
2) Extract window embeddings using pretrained HybridDL10Large.encode().
3) Aggregate to subject-level features:
   - per-subject mean/std over window embeddings
   - optional pose-wise means concatenation
4) Train small-sample classifiers (SVM/LogReg/RF), evaluate on fixed test subjects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import prepare_dl10_holdout32_step3 as prep
from train_dl10_hybrid_large_fold import HybridDL10Large


LENGTH = 160
EMB_DIM = 352  # Hybrid encode() output dim
N_POSES = 10


def load_state(ckpt_path: Path, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def build_xy(manifest: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(manifest)
    x_arr = np.zeros((n, LENGTH, 96), dtype=np.float32)
    m_arr = np.zeros((n, LENGTH), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i, row in enumerate(manifest.itertuples(index=False)):
        x = prep.read_sensor_matrix(str(row.file_path))
        x = (x - mean) / std
        xf, mk = prep.crop_or_pad(x, length=LENGTH, mode="center", rng=rng, pad_mode="zero")
        x_arr[i] = xf
        m_arr[i] = mk
    return x_arr, m_arr


@torch.no_grad()
def encode_windows(model: HybridDL10Large, x: np.ndarray, m: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for s in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).to(device)
        mb = torch.from_numpy(m[s : s + batch_size]).to(device)
        z = model.encode(xb, mb).detach().cpu().numpy()
        out.append(z)
    return np.concatenate(out, axis=0).astype(np.float64)


def subject_features(manifest: pd.DataFrame, win_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = manifest.reset_index(drop=True).copy()
    df["_i"] = np.arange(len(df))
    xs, ys, subs = [], [], []
    for sid, g in df.groupby("subject_id", sort=False):
        idx = g["_i"].to_numpy(dtype=int)
        z = win_emb[idx]  # [W,D]
        mu = z.mean(axis=0)
        sd = z.std(axis=0)
        # pose-wise means: [10,D], missing pose -> zeros
        pm = np.zeros((N_POSES, EMB_DIM), dtype=np.float64)
        for p in range(N_POSES):
            gp = g[g["pose_id"].astype(int) == p]
            if len(gp) > 0:
                pidx = gp["_i"].to_numpy(dtype=int)
                pm[p] = win_emb[pidx].mean(axis=0)
        feat = np.concatenate([mu, sd, pm.reshape(-1)], axis=0)
        xs.append(feat)
        ys.append(int(g["risk_class"].iloc[0]))
        subs.append(str(sid))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=int), subs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--test-subjects-csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=r"D:\data\xin\new\dl10_risk_transfer_classical_run")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    _ = rng  # keep deterministic placeholder

    manifest = pd.read_csv(Path(args.manifest), encoding="utf-8-sig")
    test_df = pd.read_csv(Path(args.test_subjects_csv), encoding="utf-8-sig")
    test_sub = sorted(set(test_df["subject_id"].astype(str).tolist()))
    all_sub = sorted(set(manifest["subject_id"].astype(str).tolist()))
    train_sub = sorted(set(all_sub) - set(test_sub))

    train_paths = manifest.loc[manifest["subject_id"].astype(str).isin(train_sub), "file_path"].astype(str).tolist()
    mean, std = prep.fit_zscore_stats(train_paths)
    x_np, m_np = build_xy(manifest, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDL10Large(n_classes=10, dropout=args.dropout).to(device)
    model.load_state_dict(load_state(Path(args.ckpt), device), strict=True)
    win_emb = encode_windows(model, x_np, m_np, device=device, batch_size=args.batch_size)

    x_sub, y_sub, subs = subject_features(manifest, win_emb)
    sub_to_i = {s: i for i, s in enumerate(subs)}
    tr_i = np.array([sub_to_i[s] for s in train_sub], dtype=int)
    te_i = np.array([sub_to_i[s] for s in test_sub], dtype=int)
    x_tr, x_te = x_sub[tr_i], x_sub[te_i]
    y_tr, y_te = y_sub[tr_i], y_sub[te_i]

    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        multi_class="multinomial",
                        solver="lbfgs",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=args.seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    lines = [
        "Transfer Risk Classification (Hybrid embeddings + classical classifier)",
        f"manifest: {args.manifest}",
        f"checkpoint: {args.ckpt}",
        f"train_subjects({len(train_sub)}): {train_sub}",
        f"test_subjects({len(test_sub)}): {test_sub}",
        "",
    ]

    for name, clf in models.items():
        clf.fit(x_tr, y_tr)
        yp = clf.predict(x_te)
        acc = float(accuracy_score(y_te, yp))
        mf1 = float(f1_score(y_te, yp, average="macro", zero_division=0))
        results[name] = {"accuracy": acc, "macro_f1": mf1}
        lines.extend(
            [
                f"[{name}]",
                json.dumps(results[name], ensure_ascii=False, indent=2),
                classification_report(y_te, yp, digits=4, zero_division=0),
                "",
            ]
        )

    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_dir / "subject_features.npy", x_sub)
    (out_dir / "subjects_order.json").write_text(json.dumps(subs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(text)
    print("Wrote:", out_dir / "report.txt")


if __name__ == "__main__":
    main()

