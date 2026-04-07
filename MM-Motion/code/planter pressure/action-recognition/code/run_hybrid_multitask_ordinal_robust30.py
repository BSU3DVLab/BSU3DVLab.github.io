#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import prepare_dl10_holdout32_step3 as prep
from train_dl10_hybrid_large_fold import HybridDL10Large
from train_dl10_risk_transfer_classical import build_xy, encode_windows, load_state


N_POSES = 10
EMB_DIM = 352


def sample_balanced_test(rng: np.random.Generator, pool_by_risk: Dict[str, List[str]]) -> List[str]:
    # fixed balanced split: 2 high / 3 mid / 2 low
    plan = {"高风险": 2, "中风险": 3, "低风险": 2}
    out = []
    for k, n in plan.items():
        arr = np.asarray(sorted(pool_by_risk[k]), dtype=object)
        out.extend(rng.choice(arr, size=n, replace=False).tolist())
    return sorted(out)


def summarize(v: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def ordinal_from_two_logits(p_ge1: np.ndarray, p_ge2: np.ndarray) -> np.ndarray:
    # cumulative -> class probs
    p0 = 1.0 - p_ge1
    p2 = p_ge2
    p1 = np.clip(p_ge1 - p_ge2, 0.0, 1.0)
    probs = np.stack([p0, p1, p2], axis=1)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def build_subject_features(
    df: pd.DataFrame,
    emb: np.ndarray,
    pose_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Subject-level feature = embedding stats + pose-wise embedding means + aux pose confidences.
    """
    x_list, y_list, s_list = [], [], []
    for sid, g in df.groupby("subject_id", sort=False):
        idx = g["_i"].to_numpy(dtype=int)
        z = emb[idx]
        pp = pose_proba[idx]

        mu = z.mean(axis=0)
        sd = z.std(axis=0)
        feat = [mu, sd]

        # pose-wise means and pose-confidence profile
        pose_means = np.zeros((N_POSES, EMB_DIM), dtype=np.float64)
        pose_conf = np.zeros((N_POSES,), dtype=np.float64)
        for p in range(N_POSES):
            gp = g[g["pose_id"].astype(int) == p]
            if len(gp) == 0:
                continue
            pidx = gp["_i"].to_numpy(dtype=int)
            pose_means[p] = emb[pidx].mean(axis=0)
            pprob = pose_proba[pidx].mean(axis=0)
            pose_conf[p] = float(pprob.max() - np.partition(pprob, -2)[-2])

        # overall aux confidence statistics
        margin = np.max(pp, axis=1) - np.partition(pp, -2, axis=1)[:, -2]
        entropy = -np.sum(pp * np.log(np.clip(pp, 1e-8, 1.0)), axis=1)

        feat.extend(
            [
                pose_means.reshape(-1),
                pose_conf,
                np.array(
                    [
                        float(margin.mean()),
                        float(margin.std()),
                        float(entropy.mean()),
                        float(entropy.std()),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        x_list.append(np.concatenate(feat, axis=0))
        y_list.append(int(g["risk_class"].iloc[0]))
        s_list.append(str(sid))

    return np.asarray(x_list, dtype=np.float64), np.asarray(y_list, dtype=int), s_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pose-c", type=float, default=1.0)
    ap.add_argument("--ord-c", type=float, default=0.5)
    ap.add_argument(
        "--forbid-test-subjects",
        type=str,
        default="",
        help="Comma-separated subject IDs forbidden from appearing in test split.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.manifest), encoding="utf-8-sig")
    excluded = {f"subject{i:02d}" for i in range(22, 29)} | {"subject32"}
    df = df[~df["subject_id"].astype(str).isin(excluded)].reset_index(drop=True)
    df["_i"] = np.arange(len(df))

    # risk pools for balanced sampling
    pool_by_risk: Dict[str, List[str]] = {"高风险": [], "中风险": [], "低风险": []}
    for r in df.drop_duplicates("subject_id")[["subject_id", "risk_level"]].itertuples(index=False):
        sid, rl = str(r.subject_id), str(r.risk_level)
        if rl in pool_by_risk:
            pool_by_risk[rl].append(sid)

    forbid = set()
    if str(args.forbid_test_subjects).strip():
        forbid = {s.strip() for s in str(args.forbid_test_subjects).split(",") if s.strip()}
        for k in list(pool_by_risk.keys()):
            pool_by_risk[k] = [s for s in pool_by_risk[k] if s not in forbid]
        # Ensure balanced 2/3/2 split is still feasible.
        if len(pool_by_risk["高风险"]) < 2 or len(pool_by_risk["中风险"]) < 3 or len(pool_by_risk["低风险"]) < 2:
            raise RuntimeError(
                "forbid-test-subjects makes balanced 2/3/2 sampling infeasible. "
                f"remaining sizes={ {k: len(v) for k, v in pool_by_risk.items()} }"
            )

    # shared encoder (Hybrid) one-shot embeddings
    train_paths = df["file_path"].astype(str).tolist()
    mean, std = prep.fit_zscore_stats(train_paths)
    x_np, m_np = build_xy(df, mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDL10Large(n_classes=10, dropout=args.dropout).to(device)
    model.load_state_dict(load_state(Path(args.ckpt), device), strict=True)
    emb = encode_windows(model, x_np, m_np, device=device, batch_size=args.batch_size).astype(np.float64)

    records = []
    for i in range(args.n_runs):
        seed = args.seed0 + i
        rng = np.random.default_rng(seed)
        test_sub = sample_balanced_test(rng, pool_by_risk)
        test_set = set(test_sub)
        train_sub = sorted(set(df["subject_id"].astype(str).unique().tolist()) - test_set)
        train_set = set(train_sub)

        # auxiliary action task: pose classifier on windows (train subjects only)
        tr_idx = df[df["subject_id"].astype(str).isin(train_set)]["_i"].to_numpy(dtype=int)
        xw_tr = emb[tr_idx]
        yw_pose = df.loc[tr_idx, "pose_id"].to_numpy(dtype=int)
        scaler_pose = StandardScaler()
        xw_tr_s = scaler_pose.fit_transform(xw_tr)
        pose_clf = LogisticRegression(
            max_iter=3000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=seed,
            C=args.pose_c,
        )
        pose_clf.fit(xw_tr_s, yw_pose)
        pose_proba = pose_clf.predict_proba(scaler_pose.transform(emb))  # [Nw,10]

        # compress action aux probabilities to 3 channels aligned with risk via learned projection
        # use train set to learn mapping from pose-proba to risk one-hot at window level
        y_risk_w = df.loc[tr_idx, "risk_class"].to_numpy(dtype=int)
        proj = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
            C=1.0,
        )
        proj.fit(pose_proba[tr_idx], y_risk_w)
        risk_hint = proj.predict_proba(pose_proba)  # [Nw,3]

        # subject features for ordinal risk head
        x_sub, y_sub, subs = build_subject_features(df, emb, risk_hint)
        sub_to_i = {s: j for j, s in enumerate(subs)}
        tr_i = np.array([sub_to_i[s] for s in train_sub], dtype=int)
        te_i = np.array([sub_to_i[s] for s in test_sub], dtype=int)
        x_tr, x_te = x_sub[tr_i], x_sub[te_i]
        y_tr, y_te = y_sub[tr_i], y_sub[te_i]

        # ordinal head: two binary tasks y>=1 and y>=2
        z_tr = StandardScaler().fit_transform(x_tr)
        z_te = StandardScaler().fit(x_tr).transform(x_te)  # independent scaler same fit data
        y_ge1 = (y_tr >= 1).astype(int)
        y_ge2 = (y_tr >= 2).astype(int)

        clf1 = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            C=args.ord_c,
        )
        clf2 = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed + 1,
            C=args.ord_c,
        )
        clf1.fit(z_tr, y_ge1)
        clf2.fit(z_tr, y_ge2)
        p_ge1 = clf1.predict_proba(z_te)[:, 1]
        p_ge2 = clf2.predict_proba(z_te)[:, 1]
        p_cls = ordinal_from_two_logits(p_ge1, p_ge2)
        yp = np.argmax(p_cls, axis=1)

        acc = float(accuracy_score(y_te, yp))
        mf1 = float(f1_score(y_te, yp, average="macro", zero_division=0))
        records.append(
            {
                "run": i,
                "seed": seed,
                "acc": acc,
                "macro_f1": mf1,
                "test_subjects": test_sub,
                "true_y": y_te.astype(int).tolist(),
                "pred_y": yp.astype(int).tolist(),
            }
        )

    accs = np.array([r["acc"] for r in records], dtype=float)
    f1s = np.array([r["macro_f1"] for r in records], dtype=float)
    summary = {
        "n_runs": int(args.n_runs),
        "rule": "balanced 2/3/2 test split, exclude subject22-28,32",
        "forbid_test_subjects": sorted(forbid),
        "acc": summarize(accs),
        "macro_f1": summarize(f1s),
        "pose_c": float(args.pose_c),
        "ord_c": float(args.ord_c),
    }

    (out_dir / "runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(
        "\n".join(
            [
                "Hybrid shared encoder + action auxiliary + subject ordinal risk head (30 runs)",
                f"manifest: {args.manifest}",
                f"checkpoint: {args.ckpt}",
                json.dumps(summary, ensure_ascii=False, indent=2),
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out_dir / "report.txt")


if __name__ == "__main__":
    main()

