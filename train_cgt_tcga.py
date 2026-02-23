#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import io
import math
import argparse
import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
from os.path import join



def _adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def _convert_array(text: bytes) -> np.ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    # tcga.db arrays were stored with np.save; allow_pickle=True is safe here
    return np.load(out, allow_pickle=True)

sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("ARRAY", _convert_array)


# ============================================================
#  TCGA SQLite access (minimal, Python3-safe)
# ============================================================

class TCGADataAccess:
    DB_FILE_NAME = "tcga.db"
    TABLE_CLINICAL = "clinical"
    TABLE_RNASEQ = "rnaseq"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db: Optional[sqlite3.Connection] = None
        self.connect()

    def connect(self):
        db_path = join(self.data_dir, self.DB_FILE_NAME)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Cannot find {db_path}. Put DRNet's tcga.db in {self.data_dir}/")
        self.db = sqlite3.connect(
            db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.db.execute("PRAGMA journal_mode = OFF;")
        self.db.execute("PRAGMA page_size = 16384;")

    def _create_temp_table_rowids(self, rowids: np.ndarray, name: str = "tmp_rowids") -> str:
        self.db.execute(f"DROP TABLE IF EXISTS {name};")
        self.db.execute(f"CREATE TEMP TABLE {name} (rowid INT);")
        rowids = np.asarray(rowids, dtype=np.int64).reshape(-1)
        if rowids.size > 0:
            self.db.executemany(f"INSERT INTO {name} VALUES (?);", [(int(r),) for r in rowids.tolist()])
        return name

    def _drop_temp_table(self, name: str):
        self.db.execute(f"DROP TABLE IF EXISTS {name};")

    def get_labelled_patients_rowids(self) -> np.ndarray:
        """
        Return clinical.rowid for patients that have rnaseq.
        """
        rows = self.db.execute(
            f"SELECT {self.TABLE_CLINICAL}.rowid "
            f"FROM {self.TABLE_CLINICAL} "
            f"WHERE {self.TABLE_CLINICAL}.id IN (SELECT clinical_id FROM {self.TABLE_RNASEQ}) "
            f"ORDER BY {self.TABLE_CLINICAL}.rowid;"
        ).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def load_rnaseq_and_dataset_names(self, clinical_rowids: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        For the given clinical rowids:
          returns:
            X: (N,D) rnaseq arrays (as stored in db)
            dataset_names: list[str] length N (may contain None)
            clinical_ids: list[str] length N
        """
        clinical_rowids = np.asarray(clinical_rowids, dtype=np.int64).reshape(-1)
        tmp = self._create_temp_table_rowids(clinical_rowids, name="tmp_rowids")

        # Get clinical ids + dataset_name in rowid order
        clin = self.db.execute(
            f"SELECT c.rowid, c.id, c.dataset_name "
            f"FROM {self.TABLE_CLINICAL} c "
            f"WHERE c.rowid IN (SELECT rowid FROM {tmp}) "
            f"ORDER BY c.rowid;"
        ).fetchall()

        rowid_order = [int(r[0]) for r in clin]
        clinical_ids = [str(r[1]) for r in clin]
        dataset_names = [("" if r[2] is None else str(r[2])) for r in clin]

        # Join rnaseq by clinical id and keep same ordering by clinical.rowid
        rnaseq = self.db.execute(
            f"SELECT c.rowid, r.data "
            f"FROM {self.TABLE_CLINICAL} c "
            f"JOIN {self.TABLE_RNASEQ} r ON r.clinical_id = c.id "
            f"WHERE c.rowid IN (SELECT rowid FROM {tmp}) "
            f"ORDER BY c.rowid;"
        ).fetchall()

        self._drop_temp_table(tmp)

        # Map rowid -> array, then reorder by clinical ordering
        map_r = {int(rr[0]): rr[1] for rr in rnaseq}
        X = [np.asarray(map_r[rid], dtype=np.float32) for rid in rowid_order]
        X = np.stack(X, axis=0)
        return X, dataset_names, clinical_ids


def try_load_minmax(minmax_dir: Optional[str], script_dir: str, data_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Try to load DRNet's min_val.npy / max_val.npy (used in their TCGA loader).
    Search order: --minmax_dir, script_dir, data_dir
    """
    candidates = []
    if minmax_dir:
        candidates.append(minmax_dir)
    candidates.append(script_dir)
    candidates.append(data_dir)

    for d in candidates:
        min_path = join(d, "min_val.npy")
        max_path = join(d, "max_val.npy")
        if os.path.exists(min_path) and os.path.exists(max_path):
            mn = np.load(min_path)
            mx = np.load(max_path)
            # DRNet tcga loader drops the last element (often a sentinel)
            if mn.ndim == 1 and mx.ndim == 1 and mn.shape[0] == mx.shape[0] and mn.shape[0] > 1:
                mn2 = mn[:-1].astype(np.float32)
                mx2 = mx[:-1].astype(np.float32)
                return mn2, mx2
    return None, None


# ============================================================
#  TCGA semi-synthetic benchmark generator (DRNet-style)
# ============================================================

def _softmax_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = a - np.max(a, axis=axis, keepdims=True)
    ea = np.exp(a)
    return ea / (np.sum(ea, axis=axis, keepdims=True) + 1e-12)

def _normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _cosine_sim_matrix(X: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    X: (N,D), P: (M,D) -> sim: (N,M)
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + eps)
    return Xn @ Pn.T

@dataclass
class TCGABenchmarkConfig:
    k: int = 3
    Q: int = 65
    kappa: float = 10.0
    C: float = 50.0
    seed: int = 0
    dose_mu: float = 0.65
    dose_sigma: float = 0.10
    noise_eps: float = 0.15

def build_tcga_semi_synthetic(x_std: np.ndarray, cfg: TCGABenchmarkConfig) -> Dict[str, np.ndarray]:
    """
    Produces:
      t: (N,) in {0..k-1}
      s: (N,) in [0,1]
      y: (N,) observed factual outcome at (t,s)
      y_full: (N,k,Q) outcome surface on s_grid
      s_grid: (Q,)
    Outcome semantics:
      - By default, larger is better (utility).
      - If you want "risk" (smaller is better), run script with --outcome_is_risk
        which flips planning/evals (minimization).
    """
    rng = np.random.RandomState(cfg.seed)
    N, D = x_std.shape
    k, Q = int(cfg.k), int(cfg.Q)

    s_grid = np.linspace(0.0, 1.0, Q, dtype=np.float32)
    sQ = s_grid[None, None, :]

    # Two prototype sets per treatment to induce heterogeneity via cosine similarity
    idx_p0 = rng.choice(N, size=k, replace=False)
    idx_p1 = rng.choice(N, size=k, replace=False)
    P0 = x_std[idx_p0]  # (k,D)
    P1 = x_std[idx_p1]  # (k,D)

    # Similarities (N,k)
    sim0 = _cosine_sim_matrix(x_std, P0)  # (N,k)
    sim1 = _cosine_sim_matrix(x_std, P1)  # (N,k)

    # Soft assignment between the two prototype modes, per treatment
    w = _softmax_np(np.stack([sim0, sim1], axis=-1), axis=-1)  # (N,k,2)
    d0 = w[:, :, 0].astype(np.float32)
    d1 = w[:, :, 1].astype(np.float32)

    # Dose response shape per treatment/mode: mixture of normals on s_grid
    mu_s0 = np.clip(rng.normal(loc=0.50, scale=0.12, size=(k,)), 0.05, 0.95).astype(np.float32)
    mu_s1 = np.clip(rng.normal(loc=0.50, scale=0.12, size=(k,)), 0.05, 0.95).astype(np.float32)
    sig_s0 = np.clip(np.abs(rng.normal(loc=0.12, scale=0.05, size=(k,))), 0.02, 0.35).astype(np.float32)
    sig_s1 = np.clip(np.abs(rng.normal(loc=0.12, scale=0.05, size=(k,))), 0.02, 0.35).astype(np.float32)

    pdf0 = _normal_pdf(sQ, mu_s0[None, :, None], sig_s0[None, :, None]).astype(np.float32)  # (1,k,Q)
    pdf1 = _normal_pdf(sQ, mu_s1[None, :, None], sig_s1[None, :, None]).astype(np.float32)  # (1,k,Q)

    g = d0[:, :, None] * pdf0 + d1[:, :, None] * pdf1                       # (N,k,Q)
    g = g / (np.max(g, axis=-1, keepdims=True) + 1e-12)                      # normalize peak to 1

    # Baseline "treatment potential" y_t in [0,1]
    mu_t = rng.normal(loc=0.55, scale=0.15, size=(k,)).astype(np.float32)
    sig_t = np.clip(np.abs(rng.normal(loc=0.10, scale=0.05, size=(k,))), 1e-3, 1.0).astype(np.float32)
    y_t = rng.normal(loc=mu_t[None, :], scale=sig_t[None, :], size=(N, k)).astype(np.float32)
    y_t += rng.normal(loc=0.0, scale=cfg.noise_eps, size=(N, k)).astype(np.float32)
    y_t = np.clip(y_t, 0.0, 1.0).astype(np.float32)

    # Full surface
    y_full = cfg.C * (y_t[:, :, None] * g)                                   # (N,k,Q)

    # Draw factual dose s ~ N(0.65,0.1) clipped
    s = rng.normal(loc=cfg.dose_mu, scale=cfg.dose_sigma, size=(N,)).astype(np.float32)
    s = np.clip(s, 1e-6, 1.0).astype(np.float32)

    # Interpolate dose on grid
    Qm1 = float(Q - 1)
    pos = s * Qm1
    lo = np.floor(pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, Q - 1)
    w_hi = (pos - lo).astype(np.float32)
    w_lo = 1.0 - w_hi

    # Compute g(s|x,t) at factual dose for assignment bias
    g_lo = g[np.arange(N)[:, None], np.arange(k)[None, :], lo[:, None]]
    g_hi = g[np.arange(N)[:, None], np.arange(k)[None, :], hi[:, None]]
    g_obs = (w_lo[:, None] * g_lo + w_hi[:, None] * g_hi).astype(np.float32)  # (N,k)

    # Treatment assignment with bias κ (higher outcome -> more likely chosen)
    logits = cfg.kappa * (y_t * g_obs)
    probs = _softmax_np(logits, axis=1).astype(np.float32)
    t = np.array([rng.choice(k, p=probs[i]) for i in range(N)], dtype=np.int64)

    # Observed factual y = surface at chosen t and dose s (interpolated)
    y_full_t = y_full[np.arange(N), t, :]                                     # (N,Q)
    y = (w_lo * y_full_t[np.arange(N), lo] + w_hi * y_full_t[np.arange(N), hi]).astype(np.float32)

    return {
        "x": x_std.astype(np.float32),
        "t": t,
        "s": s.astype(np.float32),
        "y": y.astype(np.float32),
        "y_full": y_full.astype(np.float32),
        "s_grid": s_grid.astype(np.float32),
    }


# ============================================================
#  Dataset / splits / scaling
# ============================================================

class TwinDataset(Dataset):
    """
    If y_full is provided, __getitem__ returns:
      (x, t, s, y_norm, y_full_row)
    where y_full_row: (k,Q) in ORIGINAL scale.
    """
    def __init__(self, x, t, s, y_norm, y_full: Optional[np.ndarray] = None):
        self.x = torch.from_numpy(x).float()
        self.t = torch.from_numpy(t).long()
        self.s = torch.from_numpy(s).float()
        self.y = torch.from_numpy(y_norm).float()
        self.y_full = None if y_full is None else torch.from_numpy(y_full).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y_full is None:
            return self.x[idx], self.t[idx], self.s[idx], self.y[idx]
        return self.x[idx], self.t[idx], self.s[idx], self.y[idx], self.y_full[idx]

def split_indices(N: int, seed: int = 0, frac_train=0.6, frac_val=0.2):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(frac_train * N)
    n_val = int(frac_val * N)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return idx_train, idx_val, idx_test

@dataclass
class Scalers:
    # x scalers unused (x already standardized); kept for API symmetry
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  Dose RBF Features
# ============================================================

class DoseRBF(nn.Module):
    def __init__(self, dim: int = 33, sigma: Optional[float] = None):
        super().__init__()
        self.dim = int(dim)
        centers = torch.linspace(0.0, 1.0, self.dim)
        self.register_buffer("centers", centers)
        if sigma is None:
            spacing = 1.0 / max(1, self.dim - 1)
            sigma = 2.56 * spacing
        self.sigma = float(sigma)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.clamp(0.0, 1.0)
        diff = (s.unsqueeze(-1) - self.centers.view(*([1] * s.ndim), -1)) / max(self.sigma, 1e-8)
        return torch.exp(-0.5 * diff * diff)


# ============================================================
#  Latent utils
# ============================================================

def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_diag_gauss_per_dim(q_mu, q_lv, p_mu, p_lv) -> torch.Tensor:
    q_var = torch.exp(q_lv)
    p_var = torch.exp(p_lv)
    return 0.5 * (p_lv - q_lv + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) - 1.0)

def free_bits_stopgrad(kl_per_dim_bxd: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        return kl_per_dim_bxd.sum(dim=-1).mean()
    kl_dim = kl_per_dim_bxd.mean(dim=0)  # (d,)
    kl_dim_eff = torch.where(
        kl_dim >= tau,
        kl_dim,
        kl_dim.detach() * 0.0 + float(tau)
    )
    return kl_dim_eff.sum()

def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (0.5 * (math.log(2.0 * math.pi) + logvar + (y - mu) ** 2 / (torch.exp(logvar) + 1e-8))).mean()


# ============================================================
#  Balance regularizers: HSIC + optional MMD
# ============================================================

def _one_hot(t: torch.Tensor, k: int) -> torch.Tensor:
    return F.one_hot(t.long().clamp(0, k-1), num_classes=int(k)).float()

def _rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((n, n), device=x.device)
    x2 = (x * x).sum(dim=1, keepdim=True)
    d2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    sigma2 = max(float(sigma) ** 2, 1e-8)
    return torch.exp(-d2 / (2.0 * sigma2))

def _median_heuristic_sigma(x: torch.Tensor, max_points: int = 512) -> float:
    n = x.shape[0]
    if n <= 1:
        return 1.0
    if n > max_points:
        idx = torch.randperm(n, device=x.device)[:max_points]
        x = x[idx]
    x2 = (x * x).sum(dim=1, keepdim=True)
    d2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    vals = d2[torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)]
    if vals.numel() == 0:
        return 1.0
    med = vals.median().item()
    sigma2 = max(med / 2.0, 1e-6)
    return float(math.sqrt(sigma2))

def hsic_rbf(u: torch.Tensor, t: torch.Tensor, k: int, sigma_u: float = -1.0, sigma_t: float = -1.0) -> torch.Tensor:
    n = u.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=u.device)
    if sigma_u < 0:
        sigma_u = _median_heuristic_sigma(u.detach())
    t_oh = _one_hot(t, k=k)
    if sigma_t < 0:
        sigma_t = 1.0
    K = _rbf_kernel(u, sigma=float(sigma_u))
    L = _rbf_kernel(t_oh, sigma=float(sigma_t))
    H = torch.eye(n, device=u.device) - (1.0 / n) * torch.ones((n, n), device=u.device)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH).sum() / ((n - 1) * (n - 1))
    return hsic.clamp_min(0.0)

def mmd_rbf_unbiased(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=x.device)
    sigma2 = max(float(sigma) ** 2, 1e-8)
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    dxx = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    dyy = (y2 + y2.t() - 2.0 * (y @ y.t())).clamp_min(0.0)
    dxy = (x2 + y2.t() - 2.0 * (x @ y.t())).clamp_min(0.0)
    kxx = torch.exp(-dxx / (2.0 * sigma2))
    kyy = torch.exp(-dyy / (2.0 * sigma2))
    kxy = torch.exp(-dxy / (2.0 * sigma2))
    kxx = kxx - torch.diag(torch.diag(kxx))
    kyy = kyy - torch.diag(torch.diag(kyy))
    term_xx = kxx.sum() / (n * (n - 1))
    term_yy = kyy.sum() / (m * (m - 1))
    term_xy = 2.0 * kxy.mean()
    return (term_xx + term_yy - term_xy)

def mmd_pairwise_groups(u: torch.Tensor, t: torch.Tensor, k: int, sigma: float = -1.0) -> torch.Tensor:
    if u.shape[0] < 4:
        return torch.tensor(0.0, device=u.device)
    if sigma < 0:
        sigma = _median_heuristic_sigma(u.detach())
    groups = []
    for tt in range(int(k)):
        idx = (t == tt).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() >= 2:
            groups.append(u[idx])
    if len(groups) < 2:
        return torch.tensor(0.0, device=u.device)
    mmds = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            mmds.append(mmd_rbf_unbiased(groups[i], groups[j], sigma=float(sigma)))
    return torch.stack(mmds).mean()


# ============================================================
#  Model blocks
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PriorU(nn.Module):
    def __init__(self, x_dim: int, u_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLP(x_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def forward(self, x):
        h = self.backbone(x)
        lv = self.logvar(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class PriorV(nn.Module):
    def __init__(self, x_dim: int, k: int, v_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_emb = nn.Embedding(k, 32)
        self.backbone = MLP(x_dim + 32, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, v_dim)
        self.logvar = nn.Linear(hidden, v_dim)

    def forward(self, x, t):
        te = self.t_emb(t.long())
        h = self.backbone(torch.cat([x, te], dim=-1))
        lv = self.logvar(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class EncoderUV(nn.Module):
    def __init__(self, x_dim: int, k: int, u_dim: int, v_dim: int, dose_rbf: Optional[DoseRBF],
                 hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_emb = nn.Embedding(k, 32)
        self.dose_rbf = dose_rbf
        rbf_dim = int(dose_rbf.dim) if dose_rbf is not None else 0
        in_dim = x_dim + 32 + 1 + rbf_dim + 1  # x + t + s + rbf(s) + y
        self.backbone = MLP(in_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.u_mu = nn.Linear(hidden, u_dim)
        self.u_lv = nn.Linear(hidden, u_dim)
        self.v_mu = nn.Linear(hidden, v_dim)
        self.v_lv = nn.Linear(hidden, v_dim)

    def forward(self, x, t, s, y_norm):
        te = self.t_emb(t.long())
        feats = [x, te, s.unsqueeze(-1)]
        if self.dose_rbf is not None:
            feats.append(self.dose_rbf(s))
        feats.append(y_norm.unsqueeze(-1))
        h = self.backbone(torch.cat(feats, dim=-1))
        u_lv = self.u_lv(h).clamp(-6.0, 6.0)
        v_lv = self.v_lv(h).clamp(-6.0, 6.0)
        return self.u_mu(h), u_lv, self.v_mu(h), v_lv

class DecoderUV(nn.Module):
    """
    Outputs Gaussian likelihood params in normalized y-space:
      mu:     (B,)
      logvar: (B,)
    """
    def __init__(self, x_dim: int, k: int, u_dim: int, v_dim: int, dose_rbf: Optional[DoseRBF],
                 hidden: int = 512, depth: int = 2, dropout: float = 0.1,
                 y_logvar_min: float = -2.5, y_logvar_max: float = 4.0):
        super().__init__()
        self.t_emb = nn.Embedding(k, 32)
        self.dose_rbf = dose_rbf
        self.y_logvar_min = float(y_logvar_min)
        self.y_logvar_max = float(y_logvar_max)

        rbf_dim = int(dose_rbf.dim) if dose_rbf is not None else 0
        self.trunk = MLP(x_dim + u_dim, hidden, hidden, depth=depth, dropout=dropout)
        head_in = hidden + v_dim + 32 + 1 + rbf_dim

        self.head_mu = MLP(head_in, hidden, 1, depth=2, dropout=dropout)
        self.head_lv = MLP(head_in, hidden, 1, depth=2, dropout=dropout)

    def forward(self, x, t, s, u, v):
        te = self.t_emb(t.long())
        h = self.trunk(torch.cat([x, u], dim=-1))
        feats = [h, v, te, s.unsqueeze(-1)]
        if self.dose_rbf is not None:
            feats.append(self.dose_rbf(s))
        hh = torch.cat(feats, dim=-1)
        mu = self.head_mu(hh).squeeze(-1)
        lv = self.head_lv(hh).squeeze(-1).clamp(self.y_logvar_min, self.y_logvar_max)
        return mu, lv


# ============================================================
#  Sampler / support mask
# ============================================================

def s_to_bin(s: np.ndarray, Q: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    s = np.clip(s, 0.0, 1.0)
    q = np.floor(s * float(Q - 1) + 1e-12).astype(np.int64)
    q = np.clip(q, 0, Q - 1)
    return q

def compute_sampler_weights(
    t: np.ndarray,
    s: np.ndarray,
    k: int,
    Q: int,
    scheme: str = "joint",
    gamma: float = 1.0,
    eps: float = 1.0,
    clip_max: float = 50.0,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    sbin = s_to_bin(s, Q=Q)

    if scheme not in ("joint", "marginal"):
        raise ValueError(f"Unknown scheme={scheme}. Use 'joint' or 'marginal'.")

    if scheme == "joint":
        joint = t * Q + sbin
        joint_counts = np.bincount(joint, minlength=k * Q).astype(np.float64)
        c = joint_counts[joint] + float(eps)
        w = (1.0 / c) ** float(gamma)
    else:
        t_counts = np.bincount(t, minlength=k).astype(np.float64)
        s_counts = np.bincount(sbin, minlength=Q).astype(np.float64)
        ct = t_counts[t] + float(eps)
        cs = s_counts[sbin] + float(eps)
        w = ((1.0 / ct) ** float(gamma)) * ((1.0 / cs) ** float(gamma))

    w = np.clip(w, 1e-8, float(clip_max)).astype(np.float64)
    return w

def build_support_mask(
    t_train: np.ndarray,
    s_train: np.ndarray,
    s_grid: np.ndarray,
    k: int,
    Q: int,
    q_low: float = 0.00,
    q_high: float = 0.95,
    min_count: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t_train = np.asarray(t_train, dtype=np.int64)
    s_train = np.asarray(s_train, dtype=np.float64)
    sbin = s_to_bin(s_train, Q=Q)

    counts = np.zeros((k, Q), dtype=np.int64)
    for tt in range(k):
        idx = np.where(t_train == tt)[0]
        if idx.size == 0:
            continue
        counts[tt] = np.bincount(sbin[idx], minlength=Q).astype(np.int64)

    qlo = np.zeros((k,), dtype=np.float64)
    qhi = np.ones((k,), dtype=np.float64)
    for tt in range(k):
        ss = s_train[t_train == tt]
        if ss.size == 0:
            qlo[tt] = 0.0
            qhi[tt] = 1.0
        else:
            qlo[tt] = float(np.quantile(ss, q_low)) if q_low > 0 else 0.0
            qhi[tt] = float(np.quantile(ss, q_high)) if q_high < 1 else 1.0
            qlo[tt] = float(np.clip(qlo[tt], 0.0, 1.0))
            qhi[tt] = float(np.clip(qhi[tt], 0.0, 1.0))
            if qhi[tt] < qlo[tt]:
                qhi[tt] = qlo[tt]

    mask = np.ones((k, Q), dtype=bool)
    for tt in range(k):
        mask[tt] = (s_grid >= qlo[tt]) & (s_grid <= qhi[tt])

    if min_count and int(min_count) > 0:
        mask = mask & (counts >= int(min_count))

    for tt in range(k):
        if not mask[tt].any():
            q_star = int(np.argmax(counts[tt])) if counts[tt].sum() > 0 else 0
            mask[tt, q_star] = True

    info = {
        "allowed_frac": float(mask.mean()),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "min_count": int(min_count),
    }
    return mask, info


# ============================================================
#  Evaluation helpers
# ============================================================

def _apply_support_mask_to_surface(y_hat: np.ndarray, support_mask_tq: Optional[np.ndarray], fill_value: float) -> np.ndarray:
    if support_mask_tq is None:
        return y_hat
    mask = np.asarray(support_mask_tq, dtype=bool)
    return np.where(mask[None, :, :], y_hat, float(fill_value))

@torch.no_grad()
def evaluate_curves(
    prior_u: PriorU,
    prior_v: PriorV,
    enc: EncoderUV,
    dec: DecoderUV,
    loader: DataLoader,
    y_full: np.ndarray,
    s_grid: np.ndarray,
    idx_subset: np.ndarray,
    scalers: Scalers,
    device: str,
    u_source: str = "posterior",
    support_mask_tq: Optional[np.ndarray] = None,
    mc_eval_u: int = 20,
    sample_v_eval: bool = True,
    dec_chunk: int = 8192,
    maximize: bool = True,   # True: higher is better; False: lower is better
) -> Dict[str, Any]:
    """
    Predictive mean/variance uses:
      E[Y] = E[mu]
      E[Y^2] = E[mu^2 + sigma_dec^2]
      Var[Y] = E[Y^2] - (E[Y])^2
    sigma_dec^2 is in ORIGINAL scale.
    """
    prior_u.eval(); prior_v.eval(); enc.eval(); dec.eval()

    k = int(y_full.shape[1])
    Q = int(y_full.shape[2])
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    mask_tq_f = None if support_mask_tq is None else np.asarray(support_mask_tq, dtype=np.float32)

    se_sum = 0.0
    n_sum = 0
    pe_sum = 0.0
    dpe_sum = 0.0
    se_in_sum = 0.0
    se_out_sum = 0.0
    n_in_sum = 0.0
    n_out_sum = 0.0
    bias2_sum = 0.0
    var_sum = 0.0

    offset = 0
    for batch in loader:
        if len(batch) == 4:
            x, t, s, y_norm = batch
        else:
            x, t, s, y_norm, _yfull_row = batch

        B = int(x.shape[0])
        cur_ids = idx_subset[offset:offset + B]
        offset += B

        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        if u_source == "posterior":
            mu_u, lv_u, _, _ = enc(x, t, s, y_norm)
        else:
            mu_u, lv_u = prior_u(x)

        M = max(1, int(mc_eval_u))

        y_sum = np.zeros((B, k, Q), dtype=np.float64)
        y_sum2 = np.zeros((B, k, Q), dtype=np.float64)

        q_chunk = max(1, int(dec_chunk // max(1, B)))
        q_chunk = min(q_chunk, Q)

        for _m in range(M):
            u_samp = reparam(mu_u, lv_u) if M > 1 else mu_u

            for tt in range(k):
                t_tt = torch.full((B,), tt, device=device, dtype=torch.long)
                mu_v_tt, lv_v_tt = prior_v(x, t_tt)
                v_tt = reparam(mu_v_tt, lv_v_tt) if (sample_v_eval and M > 1) else mu_v_tt

                for q0 in range(0, Q, q_chunk):
                    q1 = min(Q, q0 + q_chunk)
                    qc = q1 - q0

                    x_rep = x.repeat_interleave(qc, dim=0)
                    u_rep  = u_samp.repeat_interleave(qc, dim=0)
                    v_rep  = v_tt.repeat_interleave(qc, dim=0)

                    t_rep = torch.full((B * qc,), tt, device=device, dtype=torch.long)
                    s_rep = s_grid_t[q0:q1].repeat(B)

                    mu_hat_norm, lv_hat_norm = dec(x_rep, t_rep, s_rep, u_rep, v_rep)

                    # to ORIGINAL scale
                    mu_hat = (mu_hat_norm * scalers.y_std + scalers.y_mean).view(B, qc)
                    var_hat = (torch.exp(lv_hat_norm) * (scalers.y_std ** 2)).view(B, qc)

                    mu_np = mu_hat.detach().cpu().numpy().astype(np.float64)
                    var_np = var_hat.detach().cpu().numpy().astype(np.float64)

                    y_sum[:, tt, q0:q1] += mu_np
                    y_sum2[:, tt, q0:q1] += (mu_np * mu_np + var_np)

        y_mean = (y_sum / float(M)).astype(np.float64)
        y_second = (y_sum2 / float(M)).astype(np.float64)
        y_var = np.maximum(y_second - y_mean * y_mean, 0.0)

        gt = y_full[cur_ids].astype(np.float64)

        err = (y_mean - gt)
        err2 = err * err
        se_sum += float(err2.sum())
        n_sum += int(B * k * Q)

        bias2_sum += float(err2.sum())
        var_sum += float(y_var.sum())

        if mask_tq_f is not None:
            m = np.broadcast_to(mask_tq_f[None, :, :], (B, k, Q)).astype(np.float64)
            se_in_sum  += float((err2 * m).sum())
            se_out_sum += float((err2 * (1.0 - m)).sum())
            n_in_sum   += float(m.sum())
            n_out_sum  += float((1.0 - m).sum())

        # planning surface (apply support)
        fill = -1e9 if maximize else +1e9
        y_plan = _apply_support_mask_to_surface(y_mean, support_mask_tq, fill_value=fill)

        # DPE: choose best dose per t, compare to true best dose per t
        if maximize:
            gt_best_dose = gt.max(axis=2)                      # (B,k)
            shat_per_t = y_plan.argmax(axis=2).astype(np.int64)
        else:
            gt_best_dose = gt.min(axis=2)
            shat_per_t = y_plan.argmin(axis=2).astype(np.int64)

        gt_at_shat = np.take_along_axis(gt, shat_per_t[:, :, None], axis=2).squeeze(2)
        dpe_sum += float(np.sum(np.abs(gt_best_dose - gt_at_shat)))

        # PE: choose best (t,q) overall
        if maximize:
            gt_best_overall = gt_best_dose.max(axis=1)         # (B,)
            flat = y_plan.reshape(B, -1).argmax(axis=1).astype(np.int64)
        else:
            gt_best_overall = gt_best_dose.min(axis=1)
            flat = y_plan.reshape(B, -1).argmin(axis=1).astype(np.int64)

        that = (flat // Q).astype(np.int64)
        qhat = (flat % Q).astype(np.int64)
        gt_at_policy = gt[np.arange(B), that, qhat]
        pe_sum += float(np.sum(np.abs(gt_best_overall - gt_at_policy)))

    mise = se_sum / max(1, n_sum)
    dpe = dpe_sum / max(1, len(idx_subset))
    pe = pe_sum / max(1, len(idx_subset))

    out = {
        "MISE": float(mise),
        "DPE": float(dpe),
        "PE": float(pe),
        "u_source": u_source,
        "mc_eval_u": int(mc_eval_u),
        "sample_v_eval": bool(sample_v_eval),
        "Bias2": float(bias2_sum / max(1, n_sum)),
        "Var": float(var_sum / max(1, n_sum)),
    }
    if mask_tq_f is not None:
        out["MISE_in"]  = float(se_in_sum / max(1.0, n_in_sum))
        out["MISE_out"] = float(se_out_sum / max(1.0, n_out_sum))
    return out


# ============================================================
#  Risk-aware regret
# ============================================================

def _normal_pdf_torch(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

@torch.no_grad()
def evaluate_risk_aware_regret(
    prior_u: PriorU,
    prior_v: PriorV,
    enc: EncoderUV,
    dec: DecoderUV,
    loader: DataLoader,
    y_full: np.ndarray,
    s_grid: np.ndarray,
    idx_subset: np.ndarray,
    scalers: Scalers,
    device: str,
    u_source: str = "posterior",
    support_mask_tq: Optional[np.ndarray] = None,
    mc_eval_u: int = 20,
    sample_v_eval: bool = True,
    dec_chunk: int = 8192,
    risk_kind: str = "mean_var",     # "mean", "mean_var", "cvar_gauss"
    risk_lambda: float = 1.0,        # for mean_var
    risk_alpha: float = 0.10,        # for cvar_gauss
    maximize: bool = True,
) -> Dict[str, Any]:
    """
    Pick (t,q) by optimizing a risk-aware utility computed from predictive mean/std.
    Regret computed against the true best value on y_full.
    """
    prior_u.eval(); prior_v.eval(); enc.eval(); dec.eval()

    dev = torch.device(device)
    k = int(y_full.shape[1])
    Q = int(y_full.shape[2])
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    if risk_kind == "cvar_gauss":
        a = float(np.clip(risk_alpha, 1e-6, 0.5))
        z_a = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(a, device=dev)).item()
        z_a_t = torch.tensor(z_a, device=dev)
        cvar_coeff = (_normal_pdf_torch(z_a_t) / max(a, 1e-8)).item()  # phi(z_a)/alpha
    else:
        cvar_coeff = None

    regret_sum = 0.0
    true_best_sum = 0.0
    true_at_policy_sum = 0.0
    n_units = 0

    offset = 0
    for batch in loader:
        if len(batch) == 4:
            x, t, s, y_norm = batch
        else:
            x, t, s, y_norm, _ = batch

        B = int(x.shape[0])
        cur_ids = idx_subset[offset:offset + B]
        offset += B

        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        if u_source == "posterior":
            mu_u, lv_u, _, _ = enc(x, t, s, y_norm)
        else:
            mu_u, lv_u = prior_u(x)

        M = max(1, int(mc_eval_u))

        y_sum = np.zeros((B, k, Q), dtype=np.float64)
        y_sum2 = np.zeros((B, k, Q), dtype=np.float64)

        q_chunk = max(1, int(dec_chunk // max(1, B)))
        q_chunk = min(q_chunk, Q)

        for _m in range(M):
            u_samp = reparam(mu_u, lv_u) if M > 1 else mu_u

            for tt in range(k):
                t_tt = torch.full((B,), tt, device=device, dtype=torch.long)
                mu_v_tt, lv_v_tt = prior_v(x, t_tt)
                v_tt = reparam(mu_v_tt, lv_v_tt) if (sample_v_eval and M > 1) else mu_v_tt

                for q0 in range(0, Q, q_chunk):
                    q1 = min(Q, q0 + q_chunk)
                    qc = q1 - q0

                    x_rep = x.repeat_interleave(qc, dim=0)
                    u_rep  = u_samp.repeat_interleave(qc, dim=0)
                    v_rep  = v_tt.repeat_interleave(qc, dim=0)

                    t_rep = torch.full((B * qc,), tt, device=device, dtype=torch.long)
                    s_rep = s_grid_t[q0:q1].repeat(B)

                    mu_hat_norm, lv_hat_norm = dec(x_rep, t_rep, s_rep, u_rep, v_rep)

                    mu_hat = (mu_hat_norm * scalers.y_std + scalers.y_mean).view(B, qc)
                    var_hat = (torch.exp(lv_hat_norm) * (scalers.y_std ** 2)).view(B, qc)

                    mu_np = mu_hat.detach().cpu().numpy().astype(np.float64)
                    var_np = var_hat.detach().cpu().numpy().astype(np.float64)

                    y_sum[:, tt, q0:q1] += mu_np
                    y_sum2[:, tt, q0:q1] += (mu_np * mu_np + var_np)

        y_mean = (y_sum / float(M)).astype(np.float64)
        y_second = (y_sum2 / float(M)).astype(np.float64)
        y_var = np.maximum(y_second - y_mean * y_mean, 0.0)
        y_std = np.sqrt(y_var + 1e-12)

        # risk utility (always "bigger util is better"); if minimizing outcome, negate first
        if not maximize:
            y_mean_u = -y_mean
            y_std_u = y_std
        else:
            y_mean_u = y_mean
            y_std_u = y_std

        if risk_kind == "mean":
            util = y_mean_u
        elif risk_kind == "mean_var":
            util = y_mean_u - float(risk_lambda) * y_std_u
        elif risk_kind == "cvar_gauss":
            util = y_mean_u - float(cvar_coeff) * y_std_u
        else:
            raise ValueError("risk_kind must be one of: mean, mean_var, cvar_gauss")

        fill = -1e9
        util_plan = _apply_support_mask_to_surface(util, support_mask_tq, fill_value=fill)

        flat = util_plan.reshape(B, -1).argmax(axis=1).astype(np.int64)
        that = (flat // Q).astype(np.int64)
        qhat = (flat % Q).astype(np.int64)

        gt = y_full[cur_ids].astype(np.float64)

        if maximize:
            gt_best = gt.reshape(B, -1).max(axis=1)
        else:
            gt_best = gt.reshape(B, -1).min(axis=1)

        gt_at_policy = gt[np.arange(B), that, qhat]
        regret = np.abs(gt_best - gt_at_policy)

        regret_sum += float(regret.sum())
        true_best_sum += float(gt_best.sum())
        true_at_policy_sum += float(gt_at_policy.sum())
        n_units += int(B)

    return {
        "risk_kind": str(risk_kind),
        "risk_lambda": float(risk_lambda),
        "risk_alpha": float(risk_alpha),
        "u_source": str(u_source),
        "mc_eval_u": int(mc_eval_u),
        "sample_v_eval": bool(sample_v_eval),
        "RiskRegret": float(regret_sum / max(1, n_units)),
        "TrueBest": float(true_best_sum / max(1, n_units)),
        "TrueAtPolicy": float(true_at_policy_sum / max(1, n_units)),
    }


# ============================================================
#  Coverage / calibration on factual test points
# ============================================================

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _normal_quantile(alpha: float, device: torch.device) -> float:
    p = 0.5 * (1.0 + float(alpha))
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p, device=device)).item()
    return float(z)

@torch.no_grad()
def evaluate_factual_coverage(
    prior_u: PriorU,
    prior_v: PriorV,
    enc: EncoderUV,
    dec: DecoderUV,
    loader: DataLoader,
    scalers: Scalers,
    device: str,
    mc: int = 50,
    u_source: str = "posterior",
    use_posterior_v: bool = True,
    sample_v_eval: bool = True,
    eps_std: float = 1e-6,
) -> Dict[str, Any]:
    prior_u.eval(); prior_v.eval(); enc.eval(); dec.eval()
    dev = torch.device(device)

    alphas = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    zvals = {a: _normal_quantile(a, dev) for a in alphas}

    n = 0
    cover_counts = {a: 0 for a in alphas}
    pit_vals: List[float] = []
    z_resid: List[float] = []
    nll_sum = 0.0

    for batch in loader:
        if len(batch) == 4:
            x, t, s, y_norm = batch
        else:
            x, t, s, y_norm, _ = batch

        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        B = x.shape[0]

        if u_source == "posterior":
            mu_u_q, lv_u_q, mu_v_q, lv_v_q = enc(x, t, s, y_norm)
        else:
            mu_u_q, lv_u_q = prior_u(x)
            mu_v_q, lv_v_q = None, None

        if use_posterior_v and (mu_v_q is not None):
            mu_v0, lv_v0 = mu_v_q, lv_v_q
        else:
            mu_v0, lv_v0 = prior_v(x, t)

        M = max(1, int(mc))
        sum_mu = torch.zeros((B,), device=device)
        sum_second = torch.zeros((B,), device=device)

        for _ in range(M):
            u_s = reparam(mu_u_q, lv_u_q) if M > 1 else mu_u_q
            if sample_v_eval and M > 1:
                v_s = reparam(mu_v0, lv_v0)
            else:
                v_s = mu_v0

            mu_hat, lv_hat = dec(x, t, s, u_s, v_s)
            var_dec = torch.exp(lv_hat).clamp_min(0.0)

            sum_mu += mu_hat
            sum_second += (mu_hat * mu_hat + var_dec)

        mean = sum_mu / float(M)
        second = sum_second / float(M)
        var = (second - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var + eps_std)

        z = (y_norm - mean) / std
        pit = torch.distributions.Normal(0.0, 1.0).cdf(z).clamp(0.0, 1.0)
        pit_vals.extend(pit.detach().cpu().numpy().tolist())
        z_resid.extend(z.detach().cpu().numpy().tolist())

        for a in alphas:
            zc = float(zvals[a])
            lo = mean - zc * std
            hi = mean + zc * std
            covered = ((y_norm >= lo) & (y_norm <= hi)).sum().item()
            cover_counts[a] += int(covered)

        nll = 0.5 * torch.log(2.0 * math.pi * (std * std)) + 0.5 * (z * z)
        nll_sum += float(nll.sum().item())

        n += int(B)

    cover = {a: cover_counts[a] / max(1, n) for a in alphas}
    return {
        "n": int(n),
        "coverage": cover,
        "nll_norm": float(nll_sum / max(1, n)),
        "pit": np.asarray(pit_vals, dtype=np.float64),
        "z": np.asarray(z_resid, dtype=np.float64),
    }

def plot_coverage_curve(cover: Dict[float, float], outpath: str):
    alphas = sorted(list(cover.keys()))
    emp = [cover[a] for a in alphas]
    plt.figure()
    plt.plot(alphas, emp, marker="o")
    plt.plot([min(alphas), max(alphas)], [min(alphas), max(alphas)], linestyle="--")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage calibration (factual test points)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_pit_hist(pit: np.ndarray, outpath: str, bins: int = 20):
    plt.figure()
    plt.hist(pit, bins=bins)
    plt.xlabel("PIT value")
    plt.ylabel("Count")
    plt.title("PIT histogram (should be ~uniform if calibrated)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_z_hist(z: np.ndarray, outpath: str, bins: int = 40):
    plt.figure()
    plt.hist(z, bins=bins)
    plt.xlabel("Standardized residual z=(y-mean)/std")
    plt.ylabel("Count")
    plt.title("Standardized residuals (factual test points)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ============================================================
#  Latent identity drift heatmap
# ============================================================

@torch.no_grad()
def identity_drift_heatmap(
    prior_v: PriorV,
    enc: EncoderUV,
    dec: DecoderUV,
    loader: DataLoader,
    s_grid: np.ndarray,
    device: str,
    k: int,
    Q: int,
    drift_n: int = 512,
    dec_chunk: int = 8192,
) -> np.ndarray:
    prior_v.eval(); enc.eval(); dec.eval()
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    sum_drift = np.zeros((k, Q), dtype=np.float64)
    cnt_drift = 0
    seen = 0

    for batch in loader:
        if len(batch) == 4:
            x, t_f, s_f, y_f = batch
        else:
            x, t_f, s_f, y_f, _ = batch

        B = int(x.shape[0])
        if seen >= drift_n:
            break
        if seen + B > drift_n:
            keep = drift_n - seen
            x = x[:keep]; t_f = t_f[:keep]; s_f = s_f[:keep]; y_f = y_f[:keep]
            B = keep
        seen += B

        x = x.to(device, non_blocking=True)
        t_f = t_f.to(device, non_blocking=True)
        s_f = s_f.to(device, non_blocking=True)
        y_f = y_f.to(device, non_blocking=True)

        u_anchor, _lv_u, _mv, _lv = enc(x, t_f, s_f, y_f)
        u_anchor = u_anchor.detach()

        q_chunk = max(1, int(dec_chunk // max(1, B)))
        q_chunk = min(q_chunk, Q)

        for tt in range(k):
            t_tt = torch.full((B,), tt, device=device, dtype=torch.long)
            mu_v_tt, _lv_v_tt = prior_v(x, t_tt)
            v_tt = mu_v_tt

            for q0 in range(0, Q, q_chunk):
                q1 = min(Q, q0 + q_chunk)
                qc = q1 - q0

                x_rep = x.repeat_interleave(qc, dim=0)
                u_rep  = u_anchor.repeat_interleave(qc, dim=0)
                v_rep  = v_tt.repeat_interleave(qc, dim=0)

                t_rep = torch.full((B * qc,), tt, device=device, dtype=torch.long)
                s_rep = s_grid_t[q0:q1].repeat(B)

                mu_cf, _lv_cf = dec(x_rep, t_rep, s_rep, u_rep, v_rep)
                y_cf = mu_cf.contiguous()

                mu_u_cf, _lv_u_cf, _mv_cf, _lv_cf2 = enc(x_rep, t_rep, s_rep, y_cf)

                d = ((mu_u_cf - u_rep) ** 2).mean(dim=1)
                d = d.view(B, qc).mean(dim=0)
                sum_drift[tt, q0:q1] += d.detach().cpu().numpy().astype(np.float64)

        cnt_drift += B

    if cnt_drift <= 0:
        return sum_drift
    return sum_drift / float(cnt_drift)

def plot_drift_heatmap(drift: np.ndarray, outpath: str):
    plt.figure(figsize=(10, 4))
    plt.imshow(drift, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("dose index q")
    plt.ylabel("treatment index t")
    plt.title("Latent identity drift heatmap: E||mu_u_cf - u_anchor||^2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ============================================================
#  Training
# ============================================================

@dataclass
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 128
    lr: float = 2e-4
    epochs: int = 300

    u_dim: int = 32
    v_dim: int = 16
    hidden: int = 512
    depth: int = 2
    dropout: float = 0.1

    beta_kl: float = 1.0
    kl_anneal_epochs: int = 100

    free_bits_tau: float = 0.001
    free_bits_after_anneal: bool = True

    lambda_lat: float = 0.1
    n_cf: int = 2

    lambda_prior_recon: float = 0.5
    prior_recon_anneal_epochs: int = 80

    lambda_surface: float = 1.0
    surface_anneal_epochs: int = 80
    n_surface: int = 8
    surface_u_source: str = "posterior_mu"  # "posterior_mu" or "prior_mu"

    lambda_balance: float = 0.3
    balance_anneal_epochs: int = 50
    balance_kind: str = "hsic"  # "hsic" or "mmd"
    balance_on: str = "mu"
    balance_sigma: float = -1.0

    grad_clip: float = 5.0
    log_every: int = 10
    eval_every: int = 50

    eval_mc_u: int = 20
    eval_sample_v: bool = True
    eval_dec_chunk: int = 8192

def _lin_ramp(epoch: int, ramp_epochs: int, target: float) -> float:
    if target <= 0:
        return 0.0
    if ramp_epochs <= 0:
        return float(target)
    return float(target) * min(1.0, float(epoch) / float(ramp_epochs))

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_full: np.ndarray,
    s_grid: np.ndarray,
    idx_val: np.ndarray,
    scalers: Scalers,
    k: int,
    x_dim: int,
    cfg: TrainConfig,
    dose_rbf: Optional[DoseRBF],
    eval_u: str = "both",
    support_mask_tq: Optional[np.ndarray] = None,
    maximize: bool = True,
):
    device = cfg.device
    Q = int(y_full.shape[2])
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    prior_u = PriorU(x_dim=x_dim, u_dim=cfg.u_dim, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    prior_v = PriorV(x_dim=x_dim, k=k, v_dim=cfg.v_dim, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    enc = EncoderUV(x_dim=x_dim, k=k, u_dim=cfg.u_dim, v_dim=cfg.v_dim, dose_rbf=dose_rbf,
                    hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    dec = DecoderUV(x_dim=x_dim, k=k, u_dim=cfg.u_dim, v_dim=cfg.v_dim, dose_rbf=dose_rbf,
                    hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)

    opt = torch.optim.Adam(
        list(prior_u.parameters()) + list(prior_v.parameters()) + list(enc.parameters()) + list(dec.parameters()),
        lr=cfg.lr
    )

    best_val = float("inf")
    best_state = None

    def _do_eval():
        evals = []
        if eval_u in ("posterior", "both"): evals.append("posterior")
        if eval_u in ("prior", "both"): evals.append("prior")

        outs = {}
        for usrc in evals:
            outs[usrc] = evaluate_curves(
                prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
                loader=val_loader,
                y_full=y_full, s_grid=s_grid,
                idx_subset=idx_val, scalers=scalers,
                device=device, u_source=usrc,
                support_mask_tq=support_mask_tq,
                mc_eval_u=int(cfg.eval_mc_u),
                sample_v_eval=bool(cfg.eval_sample_v),
                dec_chunk=int(cfg.eval_dec_chunk),
                maximize=bool(maximize),
            )
        return outs

    for epoch in range(1, cfg.epochs + 1):
        prior_u.train(); prior_v.train(); enc.train(); dec.train()

        beta = cfg.beta_kl * min(1.0, float(epoch) / float(cfg.kl_anneal_epochs)) if cfg.kl_anneal_epochs > 0 else cfg.beta_kl
        lam_prior = _lin_ramp(epoch, cfg.prior_recon_anneal_epochs, cfg.lambda_prior_recon)
        lam_surf  = _lin_ramp(epoch, cfg.surface_anneal_epochs,    cfg.lambda_surface)
        lam_bal   = _lin_ramp(epoch, cfg.balance_anneal_epochs,    cfg.lambda_balance)

        loss_sum = recon_sum = recon_prior_sum = klu_raw_sum = klv_raw_sum = klfb_sum = lat_sum = 0.0
        surf_sum = bal_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                x, t, s, y_norm = batch
                y_full_row = None
            else:
                x, t, s, y_norm, y_full_row = batch

            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            if y_full_row is not None:
                y_full_row = y_full_row.to(device, non_blocking=True)

            # posterior (factual)
            mu_u_q, lv_u_q, mu_v_q, lv_v_q = enc(x, t, s, y_norm)
            u = reparam(mu_u_q, lv_u_q)
            v = reparam(mu_v_q, lv_v_q)

            # priors
            mu_u_p, lv_u_p = prior_u(x)
            mu_v_p, lv_v_p = prior_v(x, t)

            # factual recon: Gaussian NLL
            mu_hat, lv_hat = dec(x, t, s, u, v)
            recon = gaussian_nll(y_norm, mu_hat, lv_hat)

            # prior recon (annealed): Gaussian NLL under prior means
            mu_hat_p, lv_hat_p = dec(x, t, s, mu_u_p, mu_v_p)
            recon_prior = gaussian_nll(y_norm, mu_hat_p, lv_hat_p)

            # KL
            kl_u_bxd = kl_diag_gauss_per_dim(mu_u_q, lv_u_q, mu_u_p, lv_u_p)
            kl_v_bxd = kl_diag_gauss_per_dim(mu_v_q, lv_v_q, mu_v_p, lv_v_p)

            kl_u_raw = kl_u_bxd.sum(dim=-1).mean()
            kl_v_raw = kl_v_bxd.sum(dim=-1).mean()

            if cfg.free_bits_after_anneal and (epoch < cfg.kl_anneal_epochs):
                kl_term = (kl_u_bxd.sum(dim=-1).mean() + kl_v_bxd.sum(dim=-1).mean())
                kl_fb_val = kl_term.detach()
            else:
                kl_u_fb = free_bits_stopgrad(kl_u_bxd, cfg.free_bits_tau)
                kl_v_fb = free_bits_stopgrad(kl_v_bxd, cfg.free_bits_tau)
                kl_term = kl_u_fb + kl_v_fb
                kl_fb_val = kl_term.detach()

            # u-invariance via CF drift
            lat = torch.tensor(0.0, device=device)
            if cfg.lambda_lat > 0 and cfg.n_cf > 0:
                B = x.shape[0]
                u_anchor = mu_u_q.detach()
                for _ in range(int(cfg.n_cf)):
                    t_cf = torch.randint(low=0, high=k, size=(B,), device=device)
                    s_cf = torch.rand((B,), device=device)

                    mu_v_cf, lv_v_cf = prior_v(x, t_cf)
                    v_cf = reparam(mu_v_cf, lv_v_cf)

                    mu_cf, _lv_cf = dec(x, t_cf, s_cf, u, v_cf)
                    y_cf = mu_cf
                    mu_u_cf, _, _, _ = enc(x, t_cf, s_cf, y_cf)
                    lat = lat + F.mse_loss(mu_u_cf, u_anchor)
                lat = lat / float(cfg.n_cf)

            # balance u ⟂ t
            bal = torch.tensor(0.0, device=device)
            if lam_bal > 0:
                u_for_bal = mu_u_q if cfg.balance_on == "mu" else u
                if cfg.balance_kind == "hsic":
                    bal = hsic_rbf(u_for_bal, t, k=k, sigma_u=float(cfg.balance_sigma), sigma_t=-1.0)
                elif cfg.balance_kind == "mmd":
                    bal = mmd_pairwise_groups(u_for_bal, t, k=k, sigma=float(cfg.balance_sigma))
                else:
                    raise ValueError(f"Unknown balance_kind={cfg.balance_kind}. Use hsic or mmd.")

            # surface supervision (Gaussian NLL)
            surf = torch.tensor(0.0, device=device)
            if (lam_surf > 0) and (y_full_row is not None) and (cfg.n_surface > 0):
                B = x.shape[0]
                n_surf = int(cfg.n_surface)

                t_surf = torch.randint(low=0, high=k, size=(B, n_surf), device=device)
                q_surf = torch.randint(low=0, high=Q, size=(B, n_surf), device=device)
                s_surf = s_grid_t[q_surf]  # (B,n_surf)

                y_target = y_full_row.gather(1, t_surf.unsqueeze(-1).expand(B, n_surf, Q)).gather(
                    2, q_surf.unsqueeze(-1)
                ).squeeze(-1)

                # normalize with y scalers
                y_target_norm = (y_target - float(scalers.y_mean)) / float(scalers.y_std)

                u_s = mu_u_q if cfg.surface_u_source == "posterior_mu" else mu_u_p

                t_flat = t_surf.reshape(-1)
                s_flat = s_surf.reshape(-1)
                x_rep = x.repeat_interleave(n_surf, dim=0)
                u_rep  = u_s.repeat_interleave(n_surf, dim=0)

                mu_v_s, lv_v_s = prior_v(x_rep, t_flat)
                v_rep = mu_v_s

                mu_surf, lv_surf = dec(x_rep, t_flat, s_flat, u_rep, v_rep)
                mu_surf = mu_surf.view(B, n_surf)
                lv_surf = lv_surf.view(B, n_surf)
                surf = gaussian_nll(y_target_norm, mu_surf, lv_surf)

            loss = (
                recon
                + lam_prior * recon_prior
                + beta * kl_term
                + cfg.lambda_lat * lat
                + lam_bal * bal
                + lam_surf * surf
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(prior_u.parameters()) + list(prior_v.parameters()) + list(enc.parameters()) + list(dec.parameters()),
                    cfg.grad_clip
                )
            opt.step()

            loss_sum += float(loss.item())
            recon_sum += float(recon.item())
            recon_prior_sum += float(recon_prior.item())
            klu_raw_sum += float(kl_u_raw.item())
            klv_raw_sum += float(kl_v_raw.item())
            klfb_sum += float(kl_fb_val.item())
            lat_sum += float(lat.item())
            bal_sum += float(bal.item())
            surf_sum += float(surf.item())
            n_batches += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | "
                f"Loss={loss_sum/max(1,n_batches):.4f} "
                f"ReconNLL={recon_sum/max(1,n_batches):.4f} "
                f"ReconP_NLL={recon_prior_sum/max(1,n_batches):.4f} "
                f"KLu_raw={klu_raw_sum/max(1,n_batches):.4f} "
                f"KLv_raw={klv_raw_sum/max(1,n_batches):.4f} "
                f"KLfb_eff={klfb_sum/max(1,n_batches):.4f} "
                f"Lat(u)={lat_sum/max(1,n_batches):.6f} "
                f"BAL({cfg.balance_kind})={bal_sum/max(1,n_batches):.6f} "
                f"SURF={surf_sum/max(1,n_batches):.6f} "
                f"| ramps: prior={lam_prior:.3f} bal={lam_bal:.3f} surf={lam_surf:.3f} beta={beta:.3f}"
            )

        if epoch % cfg.eval_every == 0 or epoch == 1:
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            outs = _do_eval()
            mise_for_ckpt = outs["posterior"]["MISE"] if "posterior" in outs else outs["prior"]["MISE"]
            if mise_for_ckpt < best_val:
                best_val = mise_for_ckpt
                best_state = {
                    "prior_u": prior_u.state_dict(),
                    "prior_v": prior_v.state_dict(),
                    "enc": enc.state_dict(),
                    "dec": dec.state_dict(),
                }

            for usrc, o in outs.items():
                msg = f"  [VAL-{usrc}] MISE={o['MISE']:.6f} DPE={o['DPE']:.6f} PE={o['PE']:.6f}"
                if "MISE_in" in o:
                    msg += f" | MISE(in)={o['MISE_in']:.6f} MISE(out)={o['MISE_out']:.6f}"
                msg += f" | Bias2={o['Bias2']:.6f} Var={o['Var']:.6f} (mc_u={o.get('mc_eval_u',0)} sample_v={o.get('sample_v_eval',False)})"
                print(msg)

    if best_state is not None:
        prior_u.load_state_dict(best_state["prior_u"])
        prior_v.load_state_dict(best_state["prior_v"])
        enc.load_state_dict(best_state["enc"])
        dec.load_state_dict(best_state["dec"])

    return prior_u, prior_v, enc, dec


# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    # ---- data ----
    p.add_argument("--data_dir", type=str, default='data/tcga/')
    p.add_argument("--subsample_n", type=int, default=6000)
    p.add_argument("--tcga_num_features", type=int, default=0,
                   help="Truncate rnaseq to first N features (0 keeps all). Strongly recommended for speed.")
    p.add_argument("--minmax_dir", type=str, default='data/tcga/',
                   help="Optional dir containing DRNet min_val.npy and max_val.npy for rnaseq min-max scaling.")
    p.add_argument("--normal_keyword", type=str, default="normal",
                   help="Heuristic: dataset_name containing this substring => normal tissue.")

    # ---- benchmark ----
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--Q", type=int, default=65)
    p.add_argument("--kappa", type=float, default=10.0)
    p.add_argument("--C", type=float, default=50.0)
    p.add_argument("--dose_mu", type=float, default=0.65)
    p.add_argument("--dose_sigma", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--outcome_is_risk", action="store_true",
                   help="If set, treat y_full as 'risk' (lower is better) and switch planning/evals to minimization.")

    # ---- model/training ----
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--u_dim", type=int, default=32)
    p.add_argument("--v_dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--beta_kl", type=float, default=1.0)
    p.add_argument("--kl_anneal_epochs", type=int, default=100)

    p.add_argument("--free_bits_tau", type=float, default=0.001)
    p.add_argument("--free_bits_after_anneal", action="store_true")

    p.add_argument("--lambda_lat", type=float, default=0.1)
    p.add_argument("--n_cf", type=int, default=2)

    p.add_argument("--lambda_prior_recon", type=float, default=0.5)
    p.add_argument("--prior_recon_anneal_epochs", type=int, default=80)

    p.add_argument("--lambda_surface", type=float, default=1.0)
    p.add_argument("--surface_anneal_epochs", type=int, default=80)
    p.add_argument("--n_surface", type=int, default=8)
    p.add_argument("--surface_u_source", type=str, default="posterior_mu", choices=["posterior_mu", "prior_mu"])

    p.add_argument("--lambda_balance", type=float, default=0.01)
    p.add_argument("--balance_anneal_epochs", type=int, default=50)
    p.add_argument("--balance_kind", type=str, default="hsic", choices=["hsic", "mmd"])
    p.add_argument("--balance_on", type=str, default="mu", choices=["mu", "sample"])
    p.add_argument("--balance_sigma", type=float, default=-1.0)

    p.add_argument("--eval_u", type=str, default="both", choices=["posterior", "prior", "both"])

    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50)

    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--sampler_scheme", type=str, default="joint", choices=["joint", "marginal"])
    p.add_argument("--sampler_gamma", type=float, default=0.5)
    p.add_argument("--sampler_eps", type=float, default=1.0)
    p.add_argument("--sampler_clip_max", type=float, default=50.0)

    p.add_argument("--support_q_low", type=float, default=0.00)
    p.add_argument("--support_q_high", type=float, default=0.95)
    p.add_argument("--support_min_count", type=int, default=0)
    p.add_argument("--no_support_mask", action="store_true")

    p.add_argument("--dose_rbf_dim", type=int, default=25)
    p.add_argument("--dose_rbf_sigma", type=float, default=0.05)
    p.add_argument("--no_dose_rbf", action="store_true")

    p.add_argument("--eval_mc_u", type=int, default=20)
    p.add_argument("--eval_sample_v", action="store_true")
    p.add_argument("--eval_dec_chunk", type=int, default=8192)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- reporting ----
    p.add_argument("--outdir", type=str, default="./out_tcga")
    p.add_argument("--report_coverage", action="store_true")
    p.add_argument("--coverage_mc", type=int, default=50)
    p.add_argument("--coverage_u_source", type=str, default="posterior", choices=["posterior", "prior"])
    p.add_argument("--coverage_use_posterior_v", action="store_true")
    p.add_argument("--coverage_sample_v", action="store_true")

    p.add_argument("--report_drift", action="store_true")
    p.add_argument("--drift_n", type=int, default=512)

    p.add_argument("--report_risk_regret", action="store_true")
    p.add_argument("--risk_kind", type=str, default="mean_var", choices=["mean", "mean_var", "cvar_gauss"])
    p.add_argument("--risk_lambda", type=float, default=1.0)
    p.add_argument("--risk_alpha", type=float, default=0.10)
    p.add_argument("--risk_u_source", type=str, default="posterior", choices=["posterior", "prior"])
    p.add_argument("--risk_mc_u", type=int, default=-1)
    p.add_argument("--risk_sample_v", action="store_true")

    args = p.parse_args()
    seed_everything(args.seed)
    _ensure_dir(args.outdir)

    maximize = (not bool(args.outcome_is_risk))

    # ---- load rowids ----
    da = TCGADataAccess(args.data_dir)
    rowids_all = da.get_labelled_patients_rowids()
    N_all = int(rowids_all.shape[0])

    n = int(args.subsample_n)
    if n <= 0 or n > N_all:
        raise ValueError(f"--subsample_n must be in [1, {N_all}]")

    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(N_all, size=n, replace=False)
    rowids_sub = rowids_all[chosen]

    # ---- load rnaseq + dataset_name for normal heuristic ----
    X_raw, dataset_names, _clinical_ids = da.load_rnaseq_and_dataset_names(rowids_sub)
    if args.tcga_num_features and int(args.tcga_num_features) > 0:
        X_raw = X_raw[:, :int(args.tcga_num_features)]

    print(f"[Load] tcga rowids={N_all} | subsample_n={n} | X_raw={tuple(X_raw.shape)}")

    # ---- optional DRNet min-max scaling ----
    script_dir = os.path.dirname(os.path.realpath(__file__))
    mn, mx = try_load_minmax(args.minmax_dir, script_dir=script_dir, data_dir=args.data_dir)
    if (mn is not None) and (mx is not None) and mn.shape[0] >= X_raw.shape[1] and mx.shape[0] >= X_raw.shape[1]:
        mn = mn[:X_raw.shape[1]]
        mx = mx[:X_raw.shape[1]]
        X0 = (X_raw - mn[None, :]) / (mx[None, :] - mn[None, :] + 1e-5)
        X0 = X0.astype(np.float32)
        print("[Preprocess] applied min-max using min_val.npy/max_val.npy")
    else:
        X0 = X_raw.astype(np.float32)
        print("[Preprocess] min_val/max_val not found or mismatched; using raw rnaseq values")

    # ---- split ----
    N = int(X0.shape[0])
    idx_train, idx_val, idx_test = split_indices(N, seed=args.seed, frac_train=0.6, frac_val=0.2)

    # ---- normal-tissue heuristic & gene standardization using TRAIN normals ----
    kw = str(args.normal_keyword).lower().strip()
    names_l = [str(s).lower() for s in dataset_names]
    normal_mask = np.array([(kw in s) for s in names_l], dtype=bool)

    normal_train = normal_mask[idx_train]
    if int(normal_train.sum()) >= 10:
        X_ref = X0[idx_train][normal_train]
        print(f"[Standardize] using TRAIN normal-tissue subset: n_normal_train={int(normal_train.sum())}")
    else:
        X_ref = X0[idx_train]
        print(f"[Standardize] no/too-few normals detected (n={int(normal_train.sum())}); using ALL TRAIN samples")

    x_mean = X_ref.mean(axis=0).astype(np.float32)
    x_std = X_ref.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    X_std = ((X0 - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

    # ---- build semi-synthetic benchmark on standardized X ----
    bcfg = TCGABenchmarkConfig(
        k=int(args.k),
        Q=int(args.Q),
        kappa=float(args.kappa),
        C=float(args.C),
        seed=int(args.seed),
        dose_mu=float(args.dose_mu),
        dose_sigma=float(args.dose_sigma),
    )
    bench = build_tcga_semi_synthetic(x_std=X_std, cfg=bcfg)
    t = bench["t"]; s = bench["s"]; y = bench["y"]; y_full = bench["y_full"]; s_grid = bench["s_grid"]
    print(f"[Bench] y_full={y_full.shape} | y range [{y.min():.3f},{y.max():.3f}] | maximize={maximize}")

    # ---- y normalization (train-only) ----
    y_mean = float(y[idx_train].mean())
    y_stdv = float(y[idx_train].std() if y[idx_train].std() > 1e-6 else 1.0)
    y_norm = ((y - y_mean) / y_stdv).astype(np.float32)

    scalers = Scalers(
        x_mean=np.zeros((X_std.shape[1],), dtype=np.float32),
        x_std=np.ones((X_std.shape[1],), dtype=np.float32),
        y_mean=y_mean,
        y_std=y_stdv,
    )

    # ---- dataset objects ----
    ds_train = TwinDataset(X_std[idx_train], t[idx_train], s[idx_train], y_norm[idx_train], y_full=y_full[idx_train])
    ds_val   = TwinDataset(X_std[idx_val],   t[idx_val],   s[idx_val],   y_norm[idx_val],   y_full=None)
    ds_test  = TwinDataset(X_std[idx_test],  t[idx_test],  s[idx_test],  y_norm[idx_test],  y_full=None)

    # ---- support mask ----
    support_mask_tq = None
    if not args.no_support_mask:
        support_mask_tq, support_info = build_support_mask(
            t_train=t[idx_train],
            s_train=s[idx_train],
            s_grid=s_grid,
            k=int(args.k),
            Q=int(args.Q),
            q_low=float(args.support_q_low),
            q_high=float(args.support_q_high),
            min_count=int(args.support_min_count),
        )
        print(f"[SupportMask] enabled | allowed_frac={support_info['allowed_frac']:.3f}")
    else:
        print("[SupportMask] disabled")

    # ---- dose basis ----
    dose_rbf: Optional[DoseRBF] = None
    if not args.no_dose_rbf:
        sigma = None if float(args.dose_rbf_sigma) < 0 else float(args.dose_rbf_sigma)
        dose_rbf = DoseRBF(dim=int(args.dose_rbf_dim), sigma=sigma)

    # ---- loaders ----
    if args.use_sampler:
        w_np = compute_sampler_weights(
            t=t[idx_train], s=s[idx_train], k=int(args.k), Q=int(args.Q),
            scheme=args.sampler_scheme, gamma=args.sampler_gamma,
            eps=args.sampler_eps, clip_max=args.sampler_clip_max,
        )
        weights = torch.as_tensor(w_np, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds_train), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, shuffle=False, drop_last=False)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_loader  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        u_dim=args.u_dim,
        v_dim=args.v_dim,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        beta_kl=args.beta_kl,
        kl_anneal_epochs=args.kl_anneal_epochs,
        free_bits_tau=args.free_bits_tau,
        free_bits_after_anneal=bool(args.free_bits_after_anneal),
        lambda_lat=args.lambda_lat,
        n_cf=args.n_cf,
        lambda_prior_recon=args.lambda_prior_recon,
        prior_recon_anneal_epochs=args.prior_recon_anneal_epochs,
        lambda_surface=args.lambda_surface,
        surface_anneal_epochs=args.surface_anneal_epochs,
        n_surface=args.n_surface,
        surface_u_source=args.surface_u_source,
        lambda_balance=args.lambda_balance,
        balance_anneal_epochs=args.balance_anneal_epochs,
        balance_kind=args.balance_kind,
        balance_on=args.balance_on,
        balance_sigma=args.balance_sigma,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_mc_u=int(args.eval_mc_u),
        eval_sample_v=bool(args.eval_sample_v),
        eval_dec_chunk=int(args.eval_dec_chunk),
    )

    prior_u, prior_v, enc, dec = train(
        train_loader=train_loader,
        val_loader=val_loader,
        y_full=y_full,
        s_grid=s_grid,
        idx_val=idx_val,
        scalers=scalers,
        k=int(args.k),
        x_dim=X_std.shape[1],
        cfg=cfg,
        dose_rbf=dose_rbf,
        eval_u=args.eval_u,
        support_mask_tq=support_mask_tq,
        maximize=bool(maximize),
    )

    # ---- test eval ----
    evals = []
    if args.eval_u in ("posterior", "both"): evals.append("posterior")
    if args.eval_u in ("prior", "both"): evals.append("prior")

    for usrc in evals:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        o = evaluate_curves(
            prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            y_full=y_full, s_grid=s_grid,
            idx_subset=idx_test,
            scalers=scalers,
            device=args.device,
            u_source=usrc,
            support_mask_tq=support_mask_tq,
            mc_eval_u=int(args.eval_mc_u),
            sample_v_eval=bool(args.eval_sample_v),
            dec_chunk=int(args.eval_dec_chunk),
            maximize=bool(maximize),
        )
        msg = f"[Test-{usrc}] MISE={o['MISE']:.6f} DPE={o['DPE']:.6f} PE={o['PE']:.6f} (mc_u={o.get('mc_eval_u',0)} sample_v={o.get('sample_v_eval',False)})"
        if "MISE_in" in o:
            msg += f" | MISE(in)={o['MISE_in']:.6f} MISE(out)={o['MISE_out']:.6f}"
        print(msg)

    # ---- risk-aware regret ----
    if args.report_risk_regret:
        mc_rr = int(args.eval_mc_u) if int(args.risk_mc_u) < 0 else int(args.risk_mc_u)
        rr = evaluate_risk_aware_regret(
            prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            y_full=y_full, s_grid=s_grid,
            idx_subset=idx_test,
            scalers=scalers,
            device=args.device,
            u_source=str(args.risk_u_source),
            support_mask_tq=support_mask_tq,
            mc_eval_u=int(mc_rr),
            sample_v_eval=bool(args.risk_sample_v),
            dec_chunk=int(args.eval_dec_chunk),
            risk_kind=str(args.risk_kind),
            risk_lambda=float(args.risk_lambda),
            risk_alpha=float(args.risk_alpha),
            maximize=bool(maximize),
        )
        print(
            f"[RiskRegret-{rr['u_source']}] kind={rr['risk_kind']} "
            f"lam={rr['risk_lambda']:.3f} alpha={rr['risk_alpha']:.3f} "
            f"| RiskRegret={rr['RiskRegret']:.6f} "
            f"| TrueBest={rr['TrueBest']:.6f} TrueAtPolicy={rr['TrueAtPolicy']:.6f} "
            f"(mc_u={rr['mc_eval_u']} sample_v={rr['sample_v_eval']})"
        )

    # ---- factual coverage ----
    if args.report_coverage:
        cov = evaluate_factual_coverage(
            prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            scalers=scalers,
            device=args.device,
            mc=int(args.coverage_mc),
            u_source=str(args.coverage_u_source),
            use_posterior_v=bool(args.coverage_use_posterior_v),
            sample_v_eval=bool(args.coverage_sample_v),
        )
        print(f"[Coverage] n={cov['n']} | NLL_norm={cov['nll_norm']:.6f} | cover={cov['coverage']}")

        plot_coverage_curve(cov["coverage"], join(args.outdir, "test_coverage_curve.png"))
        plot_pit_hist(cov["pit"], join(args.outdir, "test_pit_hist.png"))
        plot_z_hist(cov["z"], join(args.outdir, "test_z_hist.png"))
        print(f"[Coverage] Saved: {join(args.outdir,'test_coverage_curve.png')}")
        print(f"[Coverage] Saved: {join(args.outdir,'test_pit_hist.png')}")
        print(f"[Coverage] Saved: {join(args.outdir,'test_z_hist.png')}")

    # ---- identity drift ----
    if args.report_drift:
        drift = identity_drift_heatmap(
            prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            s_grid=s_grid,
            device=args.device,
            k=int(args.k),
            Q=int(args.Q),
            drift_n=int(args.drift_n),
            dec_chunk=int(args.eval_dec_chunk),
        )
        outpath = join(args.outdir, "identity_drift_heatmap.png")
        plot_drift_heatmap(drift, outpath)
        np.save(join(args.outdir, "identity_drift_heatmap.npy"), drift.astype(np.float32))
        print(f"[Drift] Saved: {outpath}")
        print(f"[Drift] Saved: {join(args.outdir,'identity_drift_heatmap.npy')}")

if __name__ == "__main__":
    main()
