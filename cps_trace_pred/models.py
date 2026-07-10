"""Model wrappers for multi-label trace prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_modality_dropout_numpy(
    x: np.ndarray,
    modality_slices: Dict[str, Tuple[int, int]],
    drop_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly zero modality blocks in a copy of ``x``."""
    if drop_prob <= 0 or not modality_slices:
        return x
    out = x.copy()
    for start, end in modality_slices.values():
        mask = rng.random(out.shape[0]) < drop_prob
        out[mask, start:end] = 0.0
    return out


class LogRegOVR:
    """One-vs-rest logistic regression with standardized features."""

    def __init__(self, max_iter: int = 300) -> None:
        self.scaler = StandardScaler()
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=max_iter, n_jobs=-1))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_train_scaled = self.scaler.fit_transform(x_train)
        self.clf.fit(x_train_scaled, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(x))


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPConfig:
    hidden_dim: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 25
    train_dropout_prob: float = 0.2
    use_pos_weight: bool = True
    max_pos_weight: float = 20.0
    grad_clip_norm: float = 1.0
    seed: int = 0


class MLPBaseline:
    """2-layer MLP for multi-label classification with BCE loss."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: Optional[MLPConfig] = None,
        modality_slices: Optional[Dict[str, Tuple[int, int]]] = None,
        enable_train_modality_dropout: bool = False,
    ) -> None:
        self.cfg = cfg or MLPConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MLPHead(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)
        self.scaler = StandardScaler()
        self.modality_slices = modality_slices or {}
        self.enable_train_modality_dropout = enable_train_modality_dropout

    def _apply_train_dropout(self, xb: torch.Tensor) -> torch.Tensor:
        if not self.enable_train_modality_dropout or self.cfg.train_dropout_prob <= 0:
            return xb
        if not self.modality_slices:
            return xb
        out = xb.clone()
        for start, end in self.modality_slices.values():
            mask = (torch.rand(out.size(0), 1, device=out.device) < self.cfg.train_dropout_prob).float()
            out[:, start:end] = out[:, start:end] * (1.0 - mask)
        return out

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        set_global_seed(self.cfg.seed)
        x_train_scaled = self.scaler.fit_transform(x_train).astype(np.float32)

        x_tensor = torch.from_numpy(x_train_scaled)
        y_tensor = torch.from_numpy(y_train.astype(np.float32))
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.use_pos_weight:
            positives = y_train.sum(axis=0).astype(np.float32)
            negatives = float(y_train.shape[0]) - positives
            pos_weight = negatives / np.clip(positives, a_min=1.0, a_max=None)
            pos_weight = np.clip(pos_weight, a_min=1.0, a_max=self.cfg.max_pos_weight)
            pos_weight_t = torch.from_numpy(pos_weight).to(self.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.cfg.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                xb = self._apply_train_dropout(xb)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                optimizer.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x_scaled).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
