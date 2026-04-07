#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL10 large-capacity hybrid model (single fold):
- Wider dual TCN branches
- Larger/deeper Transformer
- FocalLoss + WeightedRandomSampler
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

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

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.mask[idx], self.y[idx]


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
            layers.append(TemporalBlock(prev, c, k=3, dilation=2 ** i, dropout=dropout))
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
        # larger TCN than baseline (96/192/192 vs 64/128/128)
        self.right = TCNBranch(in_ch=48, channels=(96, 192, 192), dropout=dropout)
        self.left = TCNBranch(in_ch=48, channels=(96, 192, 192), dropout=dropout)
        # larger transformer: d_model=256, heads=8, layers=3, ff=512
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

    def _build_stat_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        xr, xl = x[:, :, :48], x[:, :, 48:]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        r_mean = (xr * m).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        l_mean = (xl * m).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        r_var = ((xr - r_mean) * m).pow(2).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        l_var = ((xl - l_mean) * m).pow(2).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        r_std, l_std = torch.sqrt(r_var + 1e-8), torch.sqrt(l_var + 1e-8)
        xr_masked = xr.masked_fill(m <= 0, -1e9)
        xl_masked = xl.masked_fill(m <= 0, -1e9)
        r_max = xr_masked.amax(dim=(1, 2), keepdim=True)
        l_max = xl_masked.amax(dim=(1, 2), keepdim=True)
        r_energy = (xr.pow(2) * m).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        l_energy = (xl.pow(2) * m).sum(dim=(1, 2), keepdim=True) / (denom * 48.0)
        feat = torch.cat([r_mean, r_std, r_max, r_energy, l_mean, l_std, l_max, l_energy], dim=1)
        return feat.squeeze(-1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Frozen-representation vector (before final pose head): dim = 256 + 96 = 352."""
        xr, xl = x[:, :, :48], x[:, :, 48:]
        h = torch.cat([self.right(xr), self.left(xl)], dim=-1)
        h = self.fuse_proj(h)
        h = self.transformer(h, src_key_padding_mask=(mask <= 0))
        z_time = self.pool(h, mask)
        z_stat = self.stat_mlp(self._build_stat_features(x, mask))
        return torch.cat([z_time, z_stat], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, mask))


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    cnt = np.bincount(labels, minlength=n_classes).astype(np.float32)
    cnt[cnt == 0] = 1.0
    return cnt.sum() / (n_classes * cnt)


def build_sampler(labels: np.ndarray, n_classes: int) -> WeightedRandomSampler:
    cls_w = compute_class_weights(labels, n_classes=n_classes)
    sw = cls_w[labels]
    return WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(labels), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, m, y in loader:
            logits = model(x.to(device), m.to(device))
            ps.append(torch.argmax(logits, dim=1).cpu().numpy())
            ys.append(y.numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train(args) -> Tuple[Dict[str, float], str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NpzSequenceDataset(Path(args.train_npz))
    val_ds = NpzSequenceDataset(Path(args.val_npz))
    n_classes = int(max(train_ds.y.max(), val_ds.y.max()) + 1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=build_sampler(train_ds.y, n_classes=n_classes), num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HybridDL10Large(n_classes=n_classes, dropout=args.dropout).to(device)
    cls_w = torch.tensor(compute_class_weights(train_ds.y, n_classes=n_classes), dtype=torch.float32, device=device)
    criterion = FocalLoss(alpha=cls_w, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_f1, best_epoch, wait = -1.0, -1, 0
    best_state, history = None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, m), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
        scheduler.step()
        tr_loss = total_loss / len(train_ds)
        vm = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": tr_loss, **vm})
        print(
            f"Epoch {ep:03d} | loss {tr_loss:.4f} | "
            f"val_acc {vm['accuracy']:.4f} | val_bal_acc {vm['balanced_accuracy']:.4f} | val_macro_f1 {vm['macro_f1']:.4f}"
        )
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, wait = vm["macro_f1"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep}.")
                break

    if best_state is None:
        raise RuntimeError("No best state captured.")

    model.load_state_dict(best_state)
    final_m = evaluate(model, val_loader, device)

    ys, ps = [], []
    with torch.no_grad():
        for x, m, y in val_loader:
            logits = model(x.to(device), m.to(device))
            ys.append(y.numpy())
            ps.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    cls_text = classification_report(y_true, y_pred, digits=4, zero_division=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = args.run_name
    torch.save(best_state, out_dir / f"hybrid_large_{run}_best.pt")
    with open(out_dir / f"hybrid_large_{run}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    report = (
        f"DL10 Hybrid Large Report ({run})\n"
        f"train_npz: {args.train_npz}\n"
        f"val_npz: {args.val_npz}\n"
        f"best_epoch: {best_epoch}\n"
        f"epochs_ran: {len(history)}\n"
        f"final_accuracy: {final_m['accuracy']:.4f}\n"
        f"final_balanced_accuracy: {final_m['balanced_accuracy']:.4f}\n"
        f"final_macro_f1: {final_m['macro_f1']:.4f}\n\n"
        "Capacity settings:\n"
        "- TCN channels per branch: (96,192,192)\n"
        "- Transformer: d_model=256, nhead=8, layers=3, ff=512\n\n"
        f"Classification report (val):\n{cls_text}\n"
    )
    with open(out_dir / f"hybrid_large_{run}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return final_m, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", type=str, required=True)
    ap.add_argument("--val-npz", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=r"D:\data\xin\new\dl10_step6_large")
    ap.add_argument("--run-name", type=str, default="fold0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=7e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    m, _ = train(args)
    print("Training finished.")
    print(m)
    print("Saved report:", str(Path(args.out_dir) / f"hybrid_large_{args.run_name}_report.txt"))


if __name__ == "__main__":
    main()
