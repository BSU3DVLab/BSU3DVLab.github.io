#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare fixed-length inputs for DL10 holdout32 (supports optional val):

- Input: train/val/test manifests (subject-wise holdout). val is optional.
- Features: 96 raw channels (R1~R48, L1~L48)
- Normalization: z-score using TRAIN set statistics only
- Fixed length: L (default 160)
  - Train: random crop when T>L
  - Val/Test: center crop when T>L
  - T<L: zero padding + mask (or end-repeat)

Optional improvement (recommended for long sequences):
- Sliding windows: cover the full sequence with overlapping windows.
  - Train: all windows (deterministic coverage) instead of random crop
  - Val/Test: all windows, and you should evaluate by aggregating windows per file_path
    (e.g. average logits over windows belonging to the same original CSV).

Outputs (out_dir):
  - holdout32_train_mean.npy
  - holdout32_train_std.npy
  - holdout32_train.npz
  - holdout32_val.npz (if provided)
  - holdout32_test.npz
  - holdout32_train_manifest.csv
  - holdout32_val_manifest.csv (if provided)
  - holdout32_test_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


SENSOR_COLS = [f"R{i}(g)" for i in range(1, 49)] + [f"L{i}(g)" for i in range(1, 49)]


def _parse_subject_index(subject_id: str) -> Optional[int]:
    """
    subject01 -> 1
    subject1  -> 1
    """
    s = str(subject_id).strip().lower()
    if not s.startswith("subject"):
        return None
    tail = s.replace("subject", "").strip()
    if not tail.isdigit():
        return None
    return int(tail)


def load_subject_weights_kg(csv_path: str) -> Dict[str, float]:
    """
    Load weights from the user-provided CAIT risk CSV (may be GBK/GB18030 encoded).

    Expected to contain:
      - 序号 (1..N)
      - 体重（kg）

    Returns mapping like:
      subject01 -> 57.0
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    last_err = None
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc)
            last_err = None
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"Failed to read weight csv: {p} ({last_err})")

    col_idx = None
    col_w = None
    for c in df.columns:
        cs = str(c)
        if col_idx is None and ("序号" in cs or "编号" in cs):
            col_idx = c
        if col_w is None and ("体重" in cs):
            col_w = c
    if col_idx is None:
        col_idx = df.columns[1] if len(df.columns) >= 2 else df.columns[0]
    if col_w is None:
        col_w = df.columns[-1]

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        idx = r.get(col_idx)
        w = r.get(col_w)
        if pd.isna(idx) or pd.isna(w):
            continue
        try:
            i = int(float(idx))
            wk = float(w)
        except Exception:
            continue
        if wk <= 0:
            continue
        out[f"subject{i:02d}"] = wk

    if not out:
        raise RuntimeError(f"No valid weights parsed from {p}. Check columns/encoding.")
    return out


def normalize_by_weight(x_g: np.ndarray, weight_kg: float) -> np.ndarray:
    """
    x_g: [T,96] in grams-force.
    Return normalized pressure as fraction of body weight:
      x_norm = x_g / (weight_kg * 1000)
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    return x_g / (weight_kg * 1000.0)


def read_sensor_matrix(csv_path: str) -> np.ndarray:
    """Load one sample as [T, 96]. Missing channels are filled with zeros."""
    p = Path(csv_path)
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(p, encoding="utf-8")

    out = np.zeros((len(df), 96), dtype=np.float32)
    for i, col in enumerate(SENSOR_COLS):
        if col in df.columns:
            out[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(np.float32)
    return out


def fit_zscore_stats(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over all frames from training set only."""
    total_frames = 0
    sum_x = np.zeros(96, dtype=np.float64)
    sum_x2 = np.zeros(96, dtype=np.float64)

    for fp in file_paths:
        x = read_sensor_matrix(fp)  # [T, 96]
        if x.shape[0] == 0:
            continue
        sum_x += x.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_frames += x.shape[0]

    if total_frames == 0:
        raise RuntimeError("No valid frames found in training set when computing z-score stats.")

    mean = sum_x / total_frames
    var = sum_x2 / total_frames - np.square(mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def crop_or_pad(
    x: np.ndarray,
    length: int,
    mode: str,
    rng: np.random.Generator,
    pad_mode: str = "zero",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: [T, 96]
    returns:
      x_fixed: [L, 96]
      mask: [L] (1 real frame, 0 padded frame)
    """
    t = x.shape[0]
    if t >= length:
        if mode == "random":
            start = int(rng.integers(0, t - length + 1))
        elif mode == "center":
            start = (t - length) // 2
        else:
            raise ValueError(f"Unknown crop mode: {mode}")
        x_fixed = x[start : start + length]
        mask = np.ones((length,), dtype=np.float32)
        return x_fixed, mask

    x_fixed = np.zeros((length, x.shape[1]), dtype=np.float32)
    x_fixed[:t] = x
    if pad_mode == "end_repeat" and t > 0:
        x_fixed[t:] = x[t - 1]
    mask = np.zeros((length,), dtype=np.float32)
    mask[:t] = 1.0
    return x_fixed, mask


def sliding_windows(
    x: np.ndarray,
    length: int,
    stride: int,
    pad_mode: str = "zero",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one sequence x [T,96] into multiple windows.

    Returns:
      Xw:   [W, L, 96]
      Mw:   [W, L]
      starts: [W] start indices in original sequence (for trace/debug)
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    t = int(x.shape[0])
    if t <= 0:
        # keep one all-zero window
        Xw = np.zeros((1, length, x.shape[1]), dtype=np.float32)
        Mw = np.zeros((1, length), dtype=np.float32)
        return Xw, Mw, np.array([0], dtype=np.int64)

    if t <= length:
        xf, mk = crop_or_pad(x, length=length, mode="center", rng=np.random.default_rng(0), pad_mode=pad_mode)
        return xf[None, ...], mk[None, ...], np.array([0], dtype=np.int64)

    starts = list(range(0, t - length + 1, stride))
    # ensure last window covers the tail
    last = t - length
    if starts[-1] != last:
        starts.append(last)

    w = len(starts)
    Xw = np.zeros((w, length, x.shape[1]), dtype=np.float32)
    Mw = np.ones((w, length), dtype=np.float32)
    for i, s in enumerate(starts):
        Xw[i] = x[s : s + length]
    return Xw, Mw, np.array(starts, dtype=np.int64)


def build_split_arrays(
    split_df: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    length: int,
    split_name: str,
    seed: int,
    pad_mode: str,
    use_windows: bool,
    window_stride: int,
    subject_weights_kg: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Create X, mask, y, subject arrays for one split."""
    rng = np.random.default_rng(seed)
    n = len(split_df)

    # window mode: we will append variable number of windows per original file
    if use_windows:
        x_list: List[np.ndarray] = []
        m_list: List[np.ndarray] = []
        y_list: List[int] = []
        s_list: List[str] = []
        p_list: List[str] = []
        start_list: List[int] = []
    else:
        x_arr = np.zeros((n, length, 96), dtype=np.float32)
        m_arr = np.zeros((n, length), dtype=np.float32)
        y_arr = np.zeros((n,), dtype=np.int64)
        s_arr = np.empty((n,), dtype=object)
        p_arr = np.empty((n,), dtype=object)

    crop_mode = "random" if split_name == "train" else "center"

    for i, row in enumerate(split_df.itertuples(index=False)):
        x = read_sensor_matrix(row.file_path)  # [T, 96]
        if subject_weights_kg is not None:
            sid = str(row.subject_id)
            wk = subject_weights_kg.get(sid)
            if wk is None:
                idx = _parse_subject_index(sid)
                if idx is not None:
                    wk = subject_weights_kg.get(f"subject{idx:02d}")
            if wk is None:
                raise KeyError(f"Missing weight for subject_id={sid}")
            x = normalize_by_weight(x, wk)

        x = (x - mean) / std

        if use_windows:
            Xw, Mw, starts = sliding_windows(x, length=length, stride=window_stride, pad_mode=pad_mode)
            # append
            for j in range(Xw.shape[0]):
                x_list.append(Xw[j])
                m_list.append(Mw[j])
                y_list.append(int(row.pose_id))
                s_list.append(str(row.subject_id))
                p_list.append(str(row.file_path))
                start_list.append(int(starts[j]))
        else:
            x_fixed, mask = crop_or_pad(x, length=length, mode=crop_mode, rng=rng, pad_mode=pad_mode)
            x_arr[i] = x_fixed
            m_arr[i] = mask
            y_arr[i] = int(row.pose_id)
            s_arr[i] = row.subject_id
            p_arr[i] = row.file_path

    if use_windows:
        X = np.stack(x_list, axis=0).astype(np.float32)  # [Nw,L,96]
        M = np.stack(m_list, axis=0).astype(np.float32)  # [Nw,L]
        y = np.asarray(y_list, dtype=np.int64)
        s = np.asarray(s_list, dtype=object)
        p = np.asarray(p_list, dtype=object)
        st = np.asarray(start_list, dtype=np.int64)
        return {"X": X, "mask": M, "y": y, "subject_id": s, "file_path": p, "win_start": st}

    return {"X": x_arr, "mask": m_arr, "y": y_arr, "subject_id": s_arr, "file_path": p_arr}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Step-3 fixed-length inputs for DL10 holdout32.")
    ap.add_argument("--train-manifest", type=str, required=True)
    ap.add_argument("--val-manifest", type=str, default="", help="Optional validation manifest.")
    ap.add_argument("--test-manifest", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=r"D:\data\xin\new\dl10_holdout32_step3_L160")
    ap.add_argument("--length", type=int, default=160)
    ap.add_argument(
        "--weight-csv",
        type=str,
        default=r"D:\data\cait风险等级.csv",
        help="CSV containing subject weights (kg). Set to empty string to disable weight normalization.",
    )
    ap.add_argument(
        "--use-windows",
        action="store_true",
        help="If set, generate sliding windows covering full sequences (recommended).",
    )
    ap.add_argument(
        "--window-stride",
        type=int,
        default=80,
        help="Stride (in frames) for sliding windows when --use-windows is enabled.",
    )
    ap.add_argument(
        "--pad-mode",
        type=str,
        default="zero",
        choices=["zero", "end_repeat"],
        help="Padding strategy for T < L.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--file-prefix",
        type=str,
        default="holdout32",
        help="Prefix for output npz/npy/csv names (e.g. holdout40).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pfx = str(args.file_prefix).strip() or "holdout32"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(Path(args.train_manifest), encoding="utf-8-sig")
    val_df = None
    if str(args.val_manifest).strip():
        val_df = pd.read_csv(Path(args.val_manifest), encoding="utf-8-sig")
    test_df = pd.read_csv(Path(args.test_manifest), encoding="utf-8-sig")

    to_check = [("train", train_df)]
    if val_df is not None:
        to_check.append(("val", val_df))
    to_check.append(("test", test_df))

    for name, mdf in to_check:
        if not {"file_path", "subject_id", "pose_id"}.issubset(set(mdf.columns)):
            raise ValueError(f"{name} manifest must include: file_path, subject_id, pose_id")

    weights = None
    if str(args.weight_csv).strip():
        weights = load_subject_weights_kg(str(args.weight_csv))

    # z-score stats should be computed on the SAME representation that will be fed to the model.
    if weights is None:
        mean, std = fit_zscore_stats(train_df["file_path"].tolist())
    else:
        total_frames = 0
        sum_x = np.zeros(96, dtype=np.float64)
        sum_x2 = np.zeros(96, dtype=np.float64)
        for row in train_df.itertuples(index=False):
            x = read_sensor_matrix(row.file_path)
            if x.shape[0] == 0:
                continue
            sid = str(row.subject_id)
            wk = weights.get(sid)
            if wk is None:
                idx = _parse_subject_index(sid)
                if idx is not None:
                    wk = weights.get(f"subject{idx:02d}")
            if wk is None:
                raise KeyError(f"Missing weight for subject_id={sid}")
            x = normalize_by_weight(x, wk)
            sum_x += x.sum(axis=0, dtype=np.float64)
            sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total_frames += x.shape[0]
        if total_frames <= 0:
            raise RuntimeError("No valid frames found when computing z-score stats with weight normalization.")
        mean = (sum_x / total_frames).astype(np.float32)
        var = (sum_x2 / total_frames - np.square(mean)).astype(np.float32)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var).astype(np.float32)
    np.save(out_dir / f"{pfx}_train_mean.npy", mean)
    np.save(out_dir / f"{pfx}_train_std.npy", std)

    train_data = build_split_arrays(
        split_df=train_df.reset_index(drop=True),
        mean=mean,
        std=std,
        length=args.length,
        split_name="train",
        seed=args.seed,
        pad_mode=args.pad_mode,
        use_windows=bool(args.use_windows),
        window_stride=int(args.window_stride),
        subject_weights_kg=weights,
    )

    if val_df is not None:
        val_data = build_split_arrays(
            split_df=val_df.reset_index(drop=True),
            mean=mean,
            std=std,
            length=args.length,
            split_name="val",
            seed=args.seed,
            pad_mode=args.pad_mode,
            use_windows=bool(args.use_windows),
            window_stride=int(args.window_stride),
            subject_weights_kg=weights,
        )
    test_data = build_split_arrays(
        split_df=test_df.reset_index(drop=True),
        mean=mean,
        std=std,
        length=args.length,
        split_name="test",
        seed=args.seed,
        pad_mode=args.pad_mode,
        use_windows=bool(args.use_windows),
        window_stride=int(args.window_stride),
        subject_weights_kg=weights,
    )

    np.savez_compressed(out_dir / f"{pfx}_train.npz", **train_data)
    if val_df is not None:
        np.savez_compressed(out_dir / f"{pfx}_val.npz", **val_data)
    np.savez_compressed(out_dir / f"{pfx}_test.npz", **test_data)

    train_df.to_csv(out_dir / f"{pfx}_train_manifest.csv", index=False, encoding="utf-8-sig")
    if val_df is not None:
        val_df.to_csv(out_dir / f"{pfx}_val_manifest.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / f"{pfx}_test_manifest.csv", index=False, encoding="utf-8-sig")

    print("Holdout Step 3 finished.")
    print(f"L: {args.length}, pad_mode: {args.pad_mode}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Output dir: {out_dir}")
    print("Saved:")
    print(f"  - {pfx}_train_mean.npy")
    print(f"  - {pfx}_train_std.npy")
    print(f"  - {pfx}_train.npz")
    if val_df is not None:
        print(f"  - {pfx}_val.npz")
    print(f"  - {pfx}_test.npz")
    print(f"  - {pfx}_train_manifest.csv")
    if val_df is not None:
        print(f"  - {pfx}_val_manifest.csv")
    print(f"  - {pfx}_test_manifest.csv")


if __name__ == "__main__":
    main()

