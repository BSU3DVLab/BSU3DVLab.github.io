#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared helpers for stratified 2/3/2 subject holdouts and hybrid meta features."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

N_POSES = 10


def sample_balanced_test(rng: np.random.Generator, pool_by_risk: Dict[str, List[str]]) -> List[str]:
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
    p0 = 1.0 - p_ge1
    p2 = p_ge2
    p1 = np.clip(p_ge1 - p_ge2, 0.0, 1.0)
    probs = np.stack([p0, p1, p2], axis=1)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def build_subject_features(df: pd.DataFrame, emb: np.ndarray, risk_hint: np.ndarray):
    x_list, y_list, s_list = [], [], []
    for sid, g in df.groupby("subject_id", sort=False):
        idx = g["_i"].to_numpy(dtype=int)
        z = emb[idx]
        rh = risk_hint[idx]
        mu = z.mean(axis=0)
        sd = z.std(axis=0)
        pose_hint = np.zeros((N_POSES, 3), dtype=np.float64)
        pose_conf = np.zeros((N_POSES,), dtype=np.float64)
        for p in range(N_POSES):
            gp = g[g["pose_id"].astype(int) == p]
            if len(gp) == 0:
                continue
            pidx = gp["_i"].to_numpy(dtype=int)
            ph = rh[pidx].mean(axis=0)
            pose_hint[p] = ph
            pose_conf[p] = float(ph.max() - np.partition(ph, -2)[-2])
        margin = np.max(rh, axis=1) - np.partition(rh, -2, axis=1)[:, -2]
        feat = np.concatenate(
            [
                mu,
                sd,
                pose_hint.reshape(-1),
                pose_conf,
                np.array([float(margin.mean()), float(margin.std())], dtype=np.float64),
            ],
            axis=0,
        )
        x_list.append(feat)
        y_list.append(int(g["risk_class"].iloc[0]))
        s_list.append(str(sid))
    return np.asarray(x_list, dtype=np.float64), np.asarray(y_list, dtype=int), s_list
