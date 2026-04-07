#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import prepare_dl10_holdout32_step3 as prep
from run_hybrid_multitask_ordinal_robust30 import sample_balanced_test, summarize
from train_dl10_hybrid_large_fold import HybridDL10Large
from train_dl10_risk_transfer_classical import build_xy, encode_windows, load_state


N_POSES = 10


def build_subject_features_from_action_proba(
    df: pd.DataFrame,
    action_proba: np.ndarray,  # [N_trials,3]
    subject_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Subject features built from action-level risk probabilities.
    - pose-wise mean proba: [10,3]
    - pose-wise confidence margin: [10]
    - pose-wise entropy: [10]
    - pose-wise counts (num trials for that pose): [10]
    - overall mean/std of proba: [3] + [3]
    - overall mean/std of margin & entropy: [4]
    """
    df2 = df.reset_index(drop=True).copy()
    df2["_i"] = np.arange(len(df2))

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    s_list: List[str] = []

    for sid in subject_ids:
        g = df2[df2["subject_id"].astype(str) == sid]
        if len(g) == 0:
            continue
        idx = g["_i"].to_numpy(dtype=int)
        p = action_proba[idx]  # [T,3]
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        margin = np.max(p, axis=1) - np.partition(p, -2, axis=1)[:, -2]
        entropy = -np.sum(p * np.log(p), axis=1)

        pose_mean = np.zeros((N_POSES, 3), dtype=np.float64)
        pose_margin = np.zeros((N_POSES,), dtype=np.float64)
        pose_entropy = np.zeros((N_POSES,), dtype=np.float64)
        pose_cnt = np.zeros((N_POSES,), dtype=np.float64)
        for pid in range(N_POSES):
            gp = g[g["pose_id"].astype(int) == pid]
            if len(gp) == 0:
                continue
            pidx = gp["_i"].to_numpy(dtype=int)
            pp = action_proba[pidx]
            pp = np.clip(pp, 1e-8, 1.0)
            pp = pp / pp.sum(axis=1, keepdims=True)
            pose_mean[pid] = pp.mean(axis=0)
            mm = np.max(pp, axis=1) - np.partition(pp, -2, axis=1)[:, -2]
            ee = -np.sum(pp * np.log(pp), axis=1)
            pose_margin[pid] = float(mm.mean())
            pose_entropy[pid] = float(ee.mean())
            pose_cnt[pid] = float(len(gp))

        feat = np.concatenate(
            [
                pose_mean.reshape(-1),
                pose_margin,
                pose_entropy,
                pose_cnt,
                p.mean(axis=0),
                p.std(axis=0),
                np.array(
                    [float(margin.mean()), float(margin.std()), float(entropy.mean()), float(entropy.std())],
                    dtype=np.float64,
                ),
            ],
            axis=0,
        )

        x_list.append(feat)
        y_list.append(int(g["risk_class"].iloc[0]))
        s_list.append(str(sid))

    return np.asarray(x_list, dtype=np.float64), np.asarray(y_list, dtype=int), s_list


class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 3)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dp(x)
        return self.fc2(x)


def train_meta_mlp(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    seed: int,
    epochs: int = 600,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    class_weights: Tuple[float, float, float] = (1.2, 1.0, 1.2),
    return_artifacts: bool = False,
    history: Optional[List[Dict[str, Any]]] = None,
    yte: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, nn.Module, StandardScaler]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    sc = StandardScaler()
    xtr_s = sc.fit_transform(xtr).astype(np.float32)
    xte_s = sc.transform(xte).astype(np.float32)

    xt = torch.from_numpy(xtr_s)
    yt = torch.from_numpy(ytr.astype(np.int64))
    xev = torch.from_numpy(xte_s)
    yev: Optional[torch.Tensor] = None
    if yte is not None:
        yev = torch.from_numpy(np.asarray(yte, dtype=np.int64))

    model = MetaMLP(in_dim=xtr_s.shape[1], hidden=min(256, max(64, xtr_s.shape[1] // 2)), dropout=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = torch.tensor(class_weights, dtype=torch.float32)

    for ep in range(epochs):
        model.train()
        logits = model(xt)
        loss = F.cross_entropy(logits, yt, weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if history is not None:
            lr_cur = float(opt.param_groups[0]["lr"])
            model.eval()
            with torch.no_grad():
                logits_tr = model(xt)
                pred = torch.argmax(logits_tr, dim=1)
                accuracy = float((pred == yt).float().mean().item())
                row: Dict[str, Any] = {
                    "epoch": ep,
                    "loss": float(loss.detach().item()),
                    "accuracy": accuracy,
                    "learning_rate": lr_cur,
                }
                if yev is not None:
                    logits_te = model(xev)
                    val_loss = F.cross_entropy(logits_te, yev, weight=w)
                    pred_te = torch.argmax(logits_te, dim=1)
                    row["val_loss"] = float(val_loss.item())
                    row["val_accuracy"] = float((pred_te == yev).float().mean().item())
            model.train()
            history.append(row)

    model.eval()
    with torch.no_grad():
        yp = torch.argmax(model(xev), dim=1).cpu().numpy().astype(int)
    if return_artifacts:
        return yp, model.cpu(), sc
    return yp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=21000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pose-c", type=float, default=1.0)
    ap.add_argument("--action-c", type=float, default=1.0)
    ap.add_argument("--meta-c", type=float, default=1.0)
    ap.add_argument("--mlp-epochs", type=int, default=700)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.manifest), encoding="utf-8-sig")
    excluded = {f"subject{i:02d}" for i in range(22, 29)} | {"subject32"}
    df = df[~df["subject_id"].astype(str).isin(excluded)].reset_index(drop=True)
    df["_i"] = np.arange(len(df))

    pool_by_risk: Dict[str, List[str]] = {"高风险": [], "中风险": [], "低风险": []}
    for r in df.drop_duplicates("subject_id")[["subject_id", "risk_level"]].itertuples(index=False):
        sid, rl = str(r.subject_id), str(r.risk_level)
        if rl in pool_by_risk:
            pool_by_risk[rl].append(sid)

    # shared frozen action embeddings per trial (one row = one action trial csv)
    paths = df["file_path"].astype(str).tolist()
    mean, std = prep.fit_zscore_stats(paths)
    x_np, m_np = build_xy(df, mean, std)  # center crop per trial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDL10Large(n_classes=10, dropout=args.dropout).to(device)
    model.load_state_dict(load_state(Path(args.ckpt), device), strict=True)
    emb = encode_windows(model, x_np, m_np, device=device, batch_size=args.batch_size).astype(np.float64)

    all_subjects = sorted(df["subject_id"].astype(str).unique().tolist())

    records = []
    training_history: List[Dict[str, Any]] = []
    for i in range(args.n_runs):
        seed = args.seed0 + i
        rng = np.random.default_rng(seed)
        test_sub = sample_balanced_test(rng, pool_by_risk)
        test_set = set(test_sub)
        train_sub = sorted(set(all_subjects) - test_set)
        train_set = set(train_sub)

        tr_idx = df[df["subject_id"].astype(str).isin(train_set)]["_i"].to_numpy(dtype=int)
        te_idx = df[df["subject_id"].astype(str).isin(test_set)]["_i"].to_numpy(dtype=int)

        # Action-level risk model on trial embeddings -> risk proba for each trial
        sc_act = StandardScaler()
        xtr_act = sc_act.fit_transform(emb[tr_idx])
        ytr_act = df.loc[tr_idx, "risk_class"].to_numpy(dtype=int)

        act_clf = LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
            C=float(args.action_c),
        )
        act_clf.fit(xtr_act, ytr_act)
        proba_all = act_clf.predict_proba(sc_act.transform(emb))  # [N_trials,3]

        # Subject-level meta-learner on aggregated action probabilities
        x_sub, y_sub, subs = build_subject_features_from_action_proba(df, proba_all, subject_ids=all_subjects)
        s2i = {s: j for j, s in enumerate(subs)}
        tr_s = np.array([s2i[s] for s in train_sub], dtype=int)
        te_s = np.array([s2i[s] for s in test_sub], dtype=int)
        xtr_meta, xte_meta = x_sub[tr_s], x_sub[te_s]
        ytr_meta, yte_meta = y_sub[tr_s], y_sub[te_s]

        # Meta LogReg
        meta_lr = LogisticRegression(
            max_iter=8000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed + 11,
            C=float(args.meta_c),
        )
        ztr = StandardScaler().fit_transform(xtr_meta)
        zte = StandardScaler().fit(xtr_meta).transform(xte_meta)
        meta_lr.fit(ztr, ytr_meta)
        yp_lr = meta_lr.predict(zte).astype(int)

        # Meta MLP
        mlp_hist: List[Dict[str, Any]] = []
        yp_mlp = train_meta_mlp(
            xtr_meta,
            ytr_meta,
            xte_meta,
            seed=seed + 29,
            epochs=int(args.mlp_epochs),
            history=mlp_hist,
            yte=yte_meta,
        )

        rec = {
            "run": i,
            "seed": seed,
            "test_subjects": test_sub,
            "true_y": yte_meta.astype(int).tolist(),
            "pred_lr": yp_lr.astype(int).tolist(),
            "pred_mlp": yp_mlp.astype(int).tolist(),
            "acc_lr": float(accuracy_score(yte_meta, yp_lr)),
            "mf1_lr": float(f1_score(yte_meta, yp_lr, average="macro", zero_division=0)),
            "acc_mlp": float(accuracy_score(yte_meta, yp_mlp)),
            "mf1_mlp": float(f1_score(yte_meta, yp_mlp, average="macro", zero_division=0)),
        }
        records.append(rec)
        training_history.append({"run": i, "seed": seed, "meta_mlp": mlp_hist})

    acc_lr = np.array([r["acc_lr"] for r in records], dtype=float)
    f1_lr = np.array([r["mf1_lr"] for r in records], dtype=float)
    acc_mlp = np.array([r["acc_mlp"] for r in records], dtype=float)
    f1_mlp = np.array([r["mf1_mlp"] for r in records], dtype=float)

    summary = {
        "n_runs": int(args.n_runs),
        "rule": "balanced 2/3/2 test split, exclude subject22-28,32",
        "action_level": {"model": "LogReg(trial_emb->risk)", "C": float(args.action_c)},
        "meta_level": {
            "logreg": {"C": float(args.meta_c), "acc": summarize(acc_lr), "macro_f1": summarize(f1_lr)},
            "mlp": {"epochs": int(args.mlp_epochs), "acc": summarize(acc_mlp), "macro_f1": summarize(f1_mlp)},
        },
    }

    (out_dir / "runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(
        json.dumps(
            {
                "note": "meta_mlp: Keras-style per epoch — loss/accuracy on train subjects (eval, after update); val_loss/val_accuracy on holdout test subjects (weighted CE like train); learning_rate from AdamW.",
                "runs": training_history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(
        "\n".join(
            [
                "Action-level risk proba + subject meta-learner (LogReg/MLP) robust30",
                f"manifest: {args.manifest}",
                f"checkpoint: {args.ckpt}",
                f"meta MLP training curves: {out_dir / 'training_history.json'}",
                json.dumps(summary, ensure_ascii=False, indent=2),
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out_dir / "report.txt", out_dir / "training_history.json")


if __name__ == "__main__":
    main()

