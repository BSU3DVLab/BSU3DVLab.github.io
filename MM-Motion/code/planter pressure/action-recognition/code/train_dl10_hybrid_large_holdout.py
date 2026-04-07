#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL10 large-capacity hybrid model (holdout32, NO-VAL):

- Train on holdout32_train.npz
- Run fixed epochs
- Evaluate test each epoch (test-aware selection by user choice)
- Save both last checkpoint and best-by-test checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpzSequenceDataset(Dataset):
    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        self.x = d["X"].astype(np.float32)
        self.mask = d["mask"].astype(np.float32)
        self.y = d["y"].astype(np.int64)
        self.file_path = d["file_path"] if "file_path" in d.files else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self.file_path is None:
            return self.x[idx], self.mask[idx], self.y[idx]
        return self.x[idx], self.mask[idx], self.y[idx], str(self.file_path[idx])


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.short = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.drop(F.relu(self.bn2(self.conv2(out))))
        return F.relu(out + self.short(x))


class TCNBranch(nn.Module):
    def __init__(self, in_ch: int = 48, channels=(96, 192, 192), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_ch
        for i, c in enumerate(channels):
            layers.append(TemporalBlock(prev, c, k=3, dilation=2**i, dropout=dropout))
            prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.scorer(x).squeeze(-1)
        score = score.masked_fill(mask <= 0, -1e9)
        alpha = torch.softmax(score, dim=-1)
        return torch.sum(x * alpha.unsqueeze(-1), dim=1)


class HybridDL10Large(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.right = TCNBranch(in_ch=48, channels=(96, 192, 192), dropout=dropout)
        self.left = TCNBranch(in_ch=48, channels=(96, 192, 192), dropout=dropout)

        self.fuse_proj = nn.Linear(384, 256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.pool = MaskedAttentionPooling(d_model=256)

        self.stat_mlp = nn.Sequential(
            nn.Linear(8, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 96, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(192, n_classes),
        )

    def _stat_feats(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,L,96], mask [B,L]
        m = mask.unsqueeze(-1)  # [B,L,1]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1,1]
        xr, xl = x[:, :, :48], x[:, :, 48:]

        r_mean = (xr * m).sum(dim=1, keepdim=True) / denom
        l_mean = (xl * m).sum(dim=1, keepdim=True) / denom
        r_var = ((xr - r_mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        l_var = ((xl - l_mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        r_std = torch.sqrt(r_var.clamp_min(1e-8))
        l_std = torch.sqrt(l_var.clamp_min(1e-8))

        r_max = (xr.masked_fill(m <= 0, -1e9)).max(dim=1, keepdim=True).values
        l_max = (xl.masked_fill(m <= 0, -1e9)).max(dim=1, keepdim=True).values
        r_max = torch.where(r_max < -1e8, torch.zeros_like(r_max), r_max)
        l_max = torch.where(l_max < -1e8, torch.zeros_like(l_max), l_max)

        r_energy = ((xr**2) * m).sum(dim=1, keepdim=True) / denom
        l_energy = ((xl**2) * m).sum(dim=1, keepdim=True) / denom

        # scalar per-foot: mean over channels
        feats = torch.cat(
            [
                r_mean.mean(dim=2),
                r_std.mean(dim=2),
                r_max.mean(dim=2),
                r_energy.mean(dim=2),
                l_mean.mean(dim=2),
                l_std.mean(dim=2),
                l_max.mean(dim=2),
                l_energy.mean(dim=2),
            ],
            dim=1,
        )  # [B,8]
        return feats

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xr = x[:, :, :48]
        xl = x[:, :, 48:]
        fr = self.right(xr)
        fl = self.left(xl)
        f = torch.cat([fr, fl], dim=-1)  # [B,L,384]
        f = self.fuse_proj(f)  # [B,L,256]
        key_padding_mask = mask <= 0
        f = self.transformer(f, src_key_padding_mask=key_padding_mask)
        pooled = self.pool(f, mask)
        stat = self.stat_mlp(self._stat_feats(x, mask))
        z = torch.cat([pooled, stat], dim=1)
        return self.head(z)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        if len(batch) == 3:
            x, m, y = batch
        else:
            x, m, y, _fp = batch
        logits = model(x.to(device), m.to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_trial_aggregated(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    """
    If loader yields (x,mask,y,file_path), aggregate logits per file_path
    to compute trial-level metrics.
    Returns None if file_path is not available.
    """
    model.eval()
    file_to_logits: Dict[str, List[np.ndarray]] = {}
    file_to_y: Dict[str, int] = {}

    for batch in loader:
        if len(batch) != 4:
            return None
        x, m, y, fps = batch
        logits = model(x.to(device), m.to(device)).detach().cpu().numpy()
        y_np = y.numpy().astype(int)
        for i in range(len(y_np)):
            fp = str(fps[i])
            file_to_logits.setdefault(fp, []).append(logits[i])
            file_to_y[fp] = int(y_np[i])

    y_true = []
    y_pred = []
    for fp, parts in file_to_logits.items():
        avg = np.mean(np.stack(parts, axis=0), axis=0)
        y_true.append(file_to_y[fp])
        y_pred.append(int(np.argmax(avg)))
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def train(args) -> Tuple[Dict[str, float], str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NpzSequenceDataset(Path(args.train_npz))
    test_ds = NpzSequenceDataset(Path(args.test_npz))

    # sampler weights from training labels
    y = train_ds.y
    counts = np.bincount(y, minlength=args.n_classes).astype(np.float64)
    class_w = (counts.sum() / np.maximum(counts, 1.0)).astype(np.float32)
    sample_w = class_w[y]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HybridDL10Large(n_classes=args.n_classes, dropout=args.dropout).to(device)
    alpha = torch.tensor(class_w, dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=args.gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_trial_acc = -1.0
    best_epoch = 0
    best_state = None
    best_test_m: Optional[Dict[str, float]] = None
    best_test_trial_m: Optional[Dict[str, float]] = None
    model.train()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if len(batch) == 3:
                x, m, yb = batch
            else:
                x, m, yb, _fp = batch
            x = x.to(device)
            m = m.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        test_m_ep = evaluate(model, test_loader, device)
        test_trial_m_ep = evaluate_trial_aggregated(model, test_loader, device)
        score = test_trial_m_ep["accuracy"] if test_trial_m_ep is not None else test_m_ep["accuracy"]
        if score > best_trial_acc:
            best_trial_acc = float(score)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_m = test_m_ep
            best_test_trial_m = test_trial_m_ep

        rec = {
            "epoch": ep,
            "loss": avg_loss,
            "window_test_accuracy": float(test_m_ep["accuracy"]),
            "window_test_balanced_accuracy": float(test_m_ep["balanced_accuracy"]),
            "window_test_macro_f1": float(test_m_ep["macro_f1"]),
        }
        if test_trial_m_ep is not None:
            rec.update(
                {
                    "trial_test_accuracy": float(test_trial_m_ep["accuracy"]),
                    "trial_test_balanced_accuracy": float(test_trial_m_ep["balanced_accuracy"]),
                    "trial_test_macro_f1": float(test_trial_m_ep["macro_f1"]),
                }
            )
        history.append(rec)
        msg = (
            f"Epoch {ep:03d} | loss {avg_loss:.4f} | "
            f"test_acc {test_m_ep['accuracy']:.4f} | test_macro_f1 {test_m_ep['macro_f1']:.4f}"
        )
        if test_trial_m_ep is not None:
            msg += (
                f" | trial_test_acc {test_trial_m_ep['accuracy']:.4f}"
                f" | trial_test_macro_f1 {test_trial_m_ep['macro_f1']:.4f}"
            )
        print(msg)

    # Final eval uses best-by-test checkpoint if available.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = best_test_m if best_test_m is not None else evaluate(model, test_loader, device)
    test_trial_m = best_test_trial_m if best_test_trial_m is not None else evaluate_trial_aggregated(model, test_loader, device)

    # classification report (test)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, m, yb = batch
            else:
                x, m, yb, _fp = batch
            logits = model(x.to(device), m.to(device))
            ys.append(yb.numpy())
            ps.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    cls_text = classification_report(y_true, y_pred, digits=4, zero_division=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = args.run_name
    last_state = model.state_dict()
    torch.save(last_state, out_dir / f"hybrid_large_holdout_{run}_last.pt")
    if best_state is not None:
        torch.save(best_state, out_dir / f"hybrid_large_holdout_{run}_best_by_test.pt")
    with open(out_dir / f"hybrid_large_holdout_{run}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    report = (
        f"DL10 Hybrid Large Holdout Report ({run})\n"
        f"train_npz: {args.train_npz}\n"
        f"test_npz: {args.test_npz}\n"
        f"epochs_ran: {len(history)}\n"
        f"model_selection: best_by_test_trial_accuracy\n"
        f"best_epoch: {best_epoch}\n"
        f"window_test_accuracy: {test_m['accuracy']:.4f}\n"
        f"window_test_balanced_accuracy: {test_m['balanced_accuracy']:.4f}\n"
        f"window_test_macro_f1: {test_m['macro_f1']:.4f}\n"
    )
    if test_trial_m is not None:
        report += (
            f"trial_test_accuracy: {test_trial_m['accuracy']:.4f}\n"
            f"trial_test_balanced_accuracy: {test_trial_m['balanced_accuracy']:.4f}\n"
            f"trial_test_macro_f1: {test_trial_m['macro_f1']:.4f}\n"
        )
    report += (
        "\n"
        "Capacity settings:\n"
        "- TCN channels per branch: (96,192,192)\n"
        "- Transformer: d_model=256, nhead=8, layers=3, ff=512\n\n"
        f"Classification report (test):\n{cls_text}\n"
    )
    with open(out_dir / f"hybrid_large_holdout_{run}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return test_m, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", dest="train_npz", type=str, required=True)
    ap.add_argument("--test-npz", dest="test_npz", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=r"D:\data\xin\new\dl10_holdout32_step4_large")
    ap.add_argument("--run-name", type=str, default="run0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=7e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-classes", dest="n_classes", type=int, default=10)
    args = ap.parse_args()
    set_seed(args.seed)
    m, _ = train(args)
    print("Training finished.")
    print(m)
    print("Saved report:", str(Path(args.out_dir) / f"hybrid_large_holdout_{args.run_name}_report.txt"))


if __name__ == "__main__":
    main()

