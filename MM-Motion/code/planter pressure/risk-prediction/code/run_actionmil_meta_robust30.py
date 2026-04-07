#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import prepare_dl10_holdout32_step3 as prep
from run_actionrisk_meta_robust30 import build_subject_features_from_action_proba, train_meta_mlp
from run_hybrid_multitask_ordinal_robust30 import sample_balanced_test, summarize
from train_dl10_hybrid_large_fold import HybridDL10Large
from train_dl10_risk_transfer_classical import encode_windows, load_state


LENGTH = 160


def build_windows_all_trials(
    df: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    stride: int,
    pad_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows for every trial in df.
    Returns:
      Xw: [Nw, L, 96]
      Mw: [Nw, L]
      trial_index: [Nw] mapping each window to trial row index
    """
    x_list = []
    m_list = []
    t_list = []
    for i, row in enumerate(df.itertuples(index=False)):
        x = prep.read_sensor_matrix(str(row.file_path))
        x = (x - mean) / std
        Xw, Mw, _ = prep.sliding_windows(x, length=LENGTH, stride=int(stride), pad_mode=pad_mode)
        for j in range(Xw.shape[0]):
            x_list.append(Xw[j])
            m_list.append(Mw[j])
            t_list.append(i)
    X = np.stack(x_list, axis=0).astype(np.float32)
    M = np.stack(m_list, axis=0).astype(np.float32)
    T = np.asarray(t_list, dtype=np.int64)
    return X, M, T


class TrialAttentionRisk(nn.Module):
    def __init__(self, d_in: int = 352, d_att: int = 128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d_in, d_att), nn.Tanh(), nn.Linear(d_att, 1))
        self.cls = nn.Linear(d_in, 3)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        z: [B, W, D], mask: [B, W] 1 valid window else 0
        returns logits [B,3]
        """
        score = self.att(z).squeeze(-1)  # [B,W]
        score = score.masked_fill(mask <= 0, -1e9)
        a = torch.softmax(score, dim=1)  # [B,W]
        pooled = torch.sum(z * a.unsqueeze(-1), dim=1)  # [B,D]
        return self.cls(pooled)


def train_action_mil(
    z_win: np.ndarray,
    trial_index: np.ndarray,
    y_trial: np.ndarray,
    train_trials: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    w0: float,
    w1: float,
    w2: float,
    lam_margin: float,
    margin: float,
    batch_trials: int,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[TrialAttentionRisk, np.ndarray]:
    """
    Train trial-level MIL risk model on window embeddings.
    Returns trained model and trial-level probabilities for ALL trials.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrialAttentionRisk(d_in=z_win.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    cls_w = torch.tensor([w0, w1, w2], dtype=torch.float32, device=device)

    # index windows per trial
    win_by_trial: List[np.ndarray] = []
    n_trials = int(y_trial.shape[0])
    for t in range(n_trials):
        win_by_trial.append(np.where(trial_index == t)[0])

    train_trials = np.asarray(train_trials, dtype=int)

    for ep in range(int(epochs)):
        # sample a batch of trials
        bt = rng.choice(train_trials, size=min(int(batch_trials), len(train_trials)), replace=False)
        # pack to max windows
        maxw = max(int(win_by_trial[t].shape[0]) for t in bt)
        zb = np.zeros((len(bt), maxw, z_win.shape[1]), dtype=np.float32)
        mb = np.zeros((len(bt), maxw), dtype=np.float32)
        yb = np.zeros((len(bt),), dtype=np.int64)
        for i, t in enumerate(bt):
            idx = win_by_trial[t]
            zt = z_win[idx].astype(np.float32)
            zb[i, : zt.shape[0]] = zt
            mb[i, : zt.shape[0]] = 1.0
            yb[i] = int(y_trial[t])

        zb_t = torch.from_numpy(zb).to(device)
        mb_t = torch.from_numpy(mb).to(device)
        yb_t = torch.from_numpy(yb).to(device)
        logits = model(zb_t, mb_t)
        base = F.cross_entropy(logits, yb_t, weight=cls_w)
        probs = torch.softmax(logits, dim=1)

        # cost-sensitive margins: penalize true=0 if p1 > p0; true=2 if p1 > p2
        mask0 = yb_t == 0
        mask2 = yb_t == 2
        p01 = torch.tensor(0.0, device=device)
        p21 = torch.tensor(0.0, device=device)
        if mask0.any():
            p0 = probs[mask0, 0]
            p1 = probs[mask0, 1]
            p01 = F.relu(p1 - p0 + margin).mean()
        if mask2.any():
            p2 = probs[mask2, 2]
            p1 = probs[mask2, 1]
            p21 = F.relu(p1 - p2 + margin).mean()
        margin_term = float(lam_margin) * (p01 + p21)
        loss = base + margin_term

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            batch_acc = float((pred == yb_t).float().mean().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if history is not None:
            history.append(
                {
                    "epoch": ep,
                    "loss": float(loss.detach().item()),
                    "loss_ce": float(base.detach().item()),
                    "loss_margin": float(margin_term.detach().item()),
                    "batch_acc": batch_acc,
                }
            )

    # infer trial probabilities for all trials
    model.eval()
    proba_trial = np.zeros((n_trials, 3), dtype=np.float64)
    with torch.no_grad():
        for t in range(n_trials):
            idx = win_by_trial[t]
            zt = torch.from_numpy(z_win[idx].astype(np.float32)).unsqueeze(0).to(device)  # [1,W,D]
            mt = torch.ones((1, zt.shape[1]), dtype=torch.float32, device=device)
            logit = model(zt, mt)
            p = torch.softmax(logit, dim=1).cpu().numpy()[0].astype(np.float64)
            proba_trial[t] = p
    return model, proba_trial


def _dump_sklearn(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def save_submission_artifacts(
    art_dir: Path,
    mil_model: TrialAttentionRisk,
    meta_lr: LogisticRegression,
    zsc_lr: StandardScaler,
    mlp_net: torch.nn.Module,
    mlp_sc: StandardScaler,
    mean: np.ndarray,
    std: np.ndarray,
    info: Dict[str, Any],
) -> None:
    art_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mil_model.state_dict(), art_dir / "trial_attention_mil.pt")
    _dump_sklearn({"logistic_regression": meta_lr, "scaler": zsc_lr}, art_dir / "meta_logreg_bundle.joblib")
    hidden = int(mlp_net.fc1.out_features)
    in_dim = int(mlp_net.fc1.in_features)
    torch.save(
        {
            "state_dict": mlp_net.state_dict(),
            "in_dim": in_dim,
            "hidden": hidden,
            "dropout": 0.25,
        },
        art_dir / "meta_mlp_head.pt",
    )
    _dump_sklearn(mlp_sc, art_dir / "meta_mlp_scaler.joblib")
    np.save(art_dir / "window_zscore_mean.npy", mean.astype(np.float32))
    np.save(art_dir / "window_zscore_std.npy", std.astype(np.float32))
    (art_dir / "submit_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=28000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--win-stride", type=int, default=40)
    ap.add_argument("--pad-mode", type=str, default="zero")
    ap.add_argument("--mil-epochs", type=int, default=450)
    ap.add_argument("--mil-lr", type=float, default=2e-3)
    ap.add_argument("--mil-wd", type=float, default=1e-3)
    ap.add_argument("--w0", type=float, default=1.6)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.6)
    ap.add_argument("--lam-margin", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--batch-trials", type=int, default=64)
    ap.add_argument("--meta-c", type=float, default=1.0)
    ap.add_argument("--mlp-epochs", type=int, default=700)
    ap.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save MIL + meta weights for submission (requires --n-runs 1).",
    )
    args = ap.parse_args()

    if args.save_artifacts and int(args.n_runs) != 1:
        raise SystemExit("--save-artifacts requires --n-runs 1 (single holdout export).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.manifest), encoding="utf-8-sig")
    excluded = {f"subject{i:02d}" for i in range(22, 29)} | {"subject32"}
    df = df[~df["subject_id"].astype(str).isin(excluded)].reset_index(drop=True)
    df["_i"] = np.arange(len(df))

    pool_by_risk: Dict[str, List[str]] = {"高风险": [], "中风险": [], "低风险": []}
    for sid, rl in df.drop_duplicates("subject_id")[["subject_id", "risk_level"]].itertuples(index=False):
        sid, rl = str(sid), str(rl)
        if rl in pool_by_risk:
            pool_by_risk[rl].append(sid)

    all_subjects = sorted(df["subject_id"].astype(str).unique().tolist())

    # Build windows for all trials once (shared across runs)
    paths = df["file_path"].astype(str).tolist()
    mean, std = prep.fit_zscore_stats(paths)
    Xw, Mw, Tidx = build_windows_all_trials(df, mean, std, stride=int(args.win_stride), pad_mode=str(args.pad_mode))

    # Encode window embeddings once (shared across runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = HybridDL10Large(n_classes=10, dropout=float(args.dropout)).to(device)
    enc.load_state_dict(load_state(Path(args.ckpt), device), strict=True)
    z_win = encode_windows(enc, Xw, Mw, device=device, batch_size=int(args.batch_size)).astype(np.float64)  # [Nw,352]

    y_trial = df["risk_class"].to_numpy(dtype=int)

    records = []
    training_history: List[Dict[str, Any]] = []
    for i in range(int(args.n_runs)):
        seed = int(args.seed0) + i
        rng = np.random.default_rng(seed)
        test_sub = sample_balanced_test(rng, pool_by_risk)
        test_set = set(test_sub)
        train_sub = sorted(set(all_subjects) - test_set)
        train_set = set(train_sub)

        tr_trials = df[df["subject_id"].astype(str).isin(train_set)]["_i"].to_numpy(int)

        mil_hist: List[Dict[str, Any]] = []
        mil_model, proba_all = train_action_mil(
            z_win=z_win,
            trial_index=Tidx,
            y_trial=y_trial,
            train_trials=tr_trials,
            seed=seed,
            epochs=int(args.mil_epochs),
            lr=float(args.mil_lr),
            weight_decay=float(args.mil_wd),
            w0=float(args.w0),
            w1=float(args.w1),
            w2=float(args.w2),
            lam_margin=float(args.lam_margin),
            margin=float(args.margin),
            batch_trials=int(args.batch_trials),
            history=mil_hist,
        )

        # Subject-level meta-learner on aggregated action probabilities
        x_sub, y_sub, subs = build_subject_features_from_action_proba(df, proba_all, subject_ids=all_subjects)
        s2i = {s: j for j, s in enumerate(subs)}
        tr_s = np.array([s2i[s] for s in train_sub], dtype=int)
        te_s = np.array([s2i[s] for s in test_sub], dtype=int)
        xtr_meta, xte_meta = x_sub[tr_s], x_sub[te_s]
        ytr_meta, yte_meta = y_sub[tr_s], y_sub[te_s]

        meta_lr = LogisticRegression(
            max_iter=9000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed + 11,
            C=float(args.meta_c),
        )
        zsc = StandardScaler()
        ztr = zsc.fit_transform(xtr_meta)
        zte = zsc.transform(xte_meta)
        meta_lr.fit(ztr, ytr_meta)
        yp_lr = meta_lr.predict(zte).astype(int)

        mlp_hist: List[Dict[str, Any]] = []
        if args.save_artifacts:
            yp_mlp, mlp_net, mlp_sc = train_meta_mlp(
                xtr_meta,
                ytr_meta,
                xte_meta,
                seed=seed + 29,
                epochs=int(args.mlp_epochs),
                return_artifacts=True,
                history=mlp_hist,
                yte=yte_meta,
            )
        else:
            yp_mlp = train_meta_mlp(
                xtr_meta,
                ytr_meta,
                xte_meta,
                seed=seed + 29,
                epochs=int(args.mlp_epochs),
                history=mlp_hist,
                yte=yte_meta,
            )

        records.append(
            {
                "run": i,
                "seed": seed,
                "test_subjects": test_sub,
                "acc_lr": float(accuracy_score(yte_meta, yp_lr)),
                "mf1_lr": float(f1_score(yte_meta, yp_lr, average="macro", zero_division=0)),
                "acc_mlp": float(accuracy_score(yte_meta, yp_mlp)),
                "mf1_mlp": float(f1_score(yte_meta, yp_mlp, average="macro", zero_division=0)),
                "true_y": yte_meta.astype(int).tolist(),
                "pred_lr": yp_lr.astype(int).tolist(),
                "pred_mlp": yp_mlp.astype(int).tolist(),
            }
        )

        training_history.append({"run": i, "seed": seed, "mil": mil_hist, "meta_mlp": mlp_hist})

        if args.save_artifacts:
            art_dir = out_dir / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)
            info = {
                "description": "Matches actionmil_meta_robust30_newlabels_full30_v1 run 20: use --seed0 28020 --n-runs 1 (training_seed=28020).",
                "training_seed": int(seed),
                "seed0_cli": int(args.seed0),
                "n_runs": int(args.n_runs),
                "robust30_full30": {"canonical_seed0": 28000, "run_index_for_seed_28020": 20},
                "test_subjects": test_sub,
                "train_subjects": train_sub,
                "manifest": str(Path(args.manifest).resolve()),
                "backbone_checkpoint": str(Path(args.ckpt).resolve()),
                "window_length": LENGTH,
                "win_stride": int(args.win_stride),
                "pad_mode": str(args.pad_mode),
                "mil_d_in": int(z_win.shape[1]),
                "meta_logreg_C": float(args.meta_c),
                "meta_mlp_epochs": int(args.mlp_epochs),
                "acc_lr": float(records[-1]["acc_lr"]),
                "acc_mlp": float(records[-1]["acc_mlp"]),
                "artifact_files": [
                    "trial_attention_mil.pt",
                    "meta_logreg_bundle.joblib",
                    "meta_mlp_head.pt",
                    "meta_mlp_scaler.joblib",
                    "window_zscore_mean.npy",
                    "window_zscore_std.npy",
                    "training_curves.json",
                    "submit_info.json",
                ],
            }
            (art_dir / "training_curves.json").write_text(
                json.dumps({"mil": mil_hist, "meta_mlp": mlp_hist}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            save_submission_artifacts(
                art_dir,
                mil_model.cpu(),
                meta_lr,
                zsc,
                mlp_net,
                mlp_sc,
                mean,
                std,
                info,
            )
            print("Saved submission artifacts under:", art_dir.resolve())

    acc_lr = np.array([r["acc_lr"] for r in records], dtype=float)
    f1_lr = np.array([r["mf1_lr"] for r in records], dtype=float)
    acc_mlp = np.array([r["acc_mlp"] for r in records], dtype=float)
    f1_mlp = np.array([r["mf1_mlp"] for r in records], dtype=float)

    summary = {
        "n_runs": int(args.n_runs),
        "rule": "balanced 2/3/2 test split, exclude subject22-28,32",
        "action_level": {
            "model": "DL MIL attention pooling on sliding-window embeddings",
            "win_stride": int(args.win_stride),
            "mil_epochs": int(args.mil_epochs),
            "mil_lr": float(args.mil_lr),
            "mil_wd": float(args.mil_wd),
            "class_weights": [float(args.w0), float(args.w1), float(args.w2)],
            "margin_penalty": {"lambda": float(args.lam_margin), "margin": float(args.margin)},
            "batch_trials": int(args.batch_trials),
        },
        "meta_level": {
            "logreg": {"C": float(args.meta_c), "acc": summarize(acc_lr), "macro_f1": summarize(f1_lr)},
            "mlp": {"epochs": int(args.mlp_epochs), "acc": summarize(acc_mlp), "macro_f1": summarize(f1_mlp)},
        },
    }

    (out_dir / "runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(
        json.dumps(
            {
                "note_mil": "Each mil_epochs step is one random mini-batch of trials (not a full pass over all training trials). Fields: loss, loss_ce, loss_margin, batch_acc on that batch.",
                "note_meta_mlp": "Keras-style: loss/accuracy on train subjects after update (eval); val_loss/val_accuracy on 7 holdout test subjects (same weighted CE as training); learning_rate from optimizer.",
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
                "Action-level DL-MIL (attention pooling) + cost-sensitive loss + subject meta-learner robust30",
                f"manifest: {args.manifest}",
                f"checkpoint: {args.ckpt}",
                f"training curves: {out_dir / 'training_history.json'}",
                json.dumps(summary, ensure_ascii=False, indent=2),
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out_dir / "report.txt", out_dir / "training_history.json")


if __name__ == "__main__":
    main()

