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
    return np.load(out, allow_pickle=True)

sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("ARRAY", _convert_array)

class BaseDataAccess:
    pass

class DataAccess(BaseDataAccess):
    DB_FILE_NAME = "news.db"
    TABLE_NEWS = "news"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db: Optional[sqlite3.Connection] = None
        self.cache_rowid: Dict[int, tuple] = {}
        self.cache_no_rowid: Dict[int, tuple] = {}
        self.connect()

    def connect(self):
        db_path = join(self.data_dir, DataAccess.DB_FILE_NAME)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Cannot find {db_path}. Put DRNet's news.db in {self.data_dir}/")

        self.db = sqlite3.connect(
            db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.db.execute("PRAGMA journal_mode = OFF;")
        self.db.execute("PRAGMA page_size = 16384;")

    def get_num_rows(self, table_name):
        return self.db.execute(f"SELECT MAX(_ROWID_) FROM {table_name} LIMIT 1;").fetchone()[0]

    def get_row(self, table_name, rid, with_rowid=False):
        cache = self.cache_rowid if with_rowid else self.cache_no_rowid
        if rid in cache:
            return cache[rid]
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns
        query = f"SELECT {columns} FROM {table_name} WHERE rowid = ?;"
        return_value = self.db.execute(query, (rid,)).fetchone()
        cache[rid] = return_value
        return return_value

    def get_labelled_patients(self):
        return np.arange(self.get_num_rows(DataAccess.TABLE_NEWS)) + 1  # rowids 1-indexed

    def get_entry_with_id(self, rid):
        news = self.get_row(DataAccess.TABLE_NEWS, rid, with_rowid=True)
        patient_id = news[0]
        return patient_id, {"id": patient_id, "x": news[1], "z": news[2]}

def load_news_xz_drnet(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    da = DataAccess(data_dir)
    ids = da.get_labelled_patients()
    xs, zs = [], []
    for rid in ids:
        _, row = da.get_entry_with_id(int(rid))
        xs.append(np.asarray(row["x"], dtype=np.float32))
        zs.append(np.asarray(row["z"], dtype=np.float32))
    x = np.stack(xs, axis=0)
    z = np.stack(zs, axis=0)
    return x, z



def _softmax_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = a - np.max(a, axis=axis, keepdims=True)
    ea = np.exp(a)
    return ea / (np.sum(ea, axis=axis, keepdims=True) + 1e-12)

def _normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@dataclass
class NewsBenchmarkConfig:
    k: int = 16
    Q: int = 65
    kappa: float = 7.0
    beta: float = 0.25
    C: float = 50.0
    noise_eps: float = 0.15
    seed: int = 0

def build_news_semi_synthetic(x: np.ndarray, z: np.ndarray, cfg: NewsBenchmarkConfig) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(cfg.seed)
    N, Dz = z.shape
    k, Q = cfg.k, cfg.Q

    s_grid = np.linspace(0.0, 1.0, Q, dtype=np.float32)

    idx_s0 = rng.choice(N, size=k, replace=False)
    idx_s1 = rng.choice(N, size=k, replace=False)
    z_s0 = z[idx_s0]
    z_s1 = z[idx_s1]

    mu_t = rng.normal(loc=0.45, scale=0.15, size=(k,)).astype(np.float32)
    sig_t = np.abs(rng.normal(loc=0.10, scale=0.05, size=(k,))).astype(np.float32)
    sig_t = np.clip(sig_t, 1e-3, 1.0)

    mu_s0 = np.clip(rng.normal(loc=0.45, scale=0.15, size=(k,)), 0.01, 0.99).astype(np.float32)
    mu_s1 = np.clip(rng.normal(loc=0.45, scale=0.15, size=(k,)), 0.01, 0.99).astype(np.float32)
    sig_s0 = np.abs(rng.normal(loc=0.10, scale=0.05, size=(k,))).astype(np.float32)
    sig_s1 = np.abs(rng.normal(loc=0.10, scale=0.05, size=(k,))).astype(np.float32)
    sig_s0 = np.clip(sig_s0, 0.02, 0.30).astype(np.float32)
    sig_s1 = np.clip(sig_s1, 0.02, 0.30).astype(np.float32)

    y_t = rng.normal(loc=mu_t[None, :], scale=sig_t[None, :], size=(N, k)).astype(np.float32)
    y_t += rng.normal(loc=0.0, scale=cfg.noise_eps, size=(N, k)).astype(np.float32)
    y_t = np.clip(y_t, 0.0, 1.0).astype(np.float32)

    dist0 = np.linalg.norm(z[:, None, :] - z_s0[None, :, :], axis=-1)  # (N,k)
    dist1 = np.linalg.norm(z[:, None, :] - z_s1[None, :, :], axis=-1)  # (N,k)
    w = _softmax_np(np.stack([-dist0, -dist1], axis=-1), axis=-1)       # (N,k,2)
    d0 = w[:, :, 0].astype(np.float32)
    d1 = w[:, :, 1].astype(np.float32)

    sQ = s_grid[None, None, :]
    pdf0 = _normal_pdf(sQ, mu_s0[None, :, None], sig_s0[None, :, None]).astype(np.float32)
    pdf1 = _normal_pdf(sQ, mu_s1[None, :, None], sig_s1[None, :, None]).astype(np.float32)

    g = d0[:, :, None] * pdf0 + d1[:, :, None] * pdf1
    g = g / (np.max(g, axis=-1, keepdims=True) + 1e-12)

    y_full = cfg.C * (y_t[:, :, None] * g)

    raw = rng.exponential(scale=cfg.beta, size=(N,)).astype(np.float32)
    s = np.clip(raw, 1e-6, 1.0).astype(np.float32)

    Qm1 = float(Q - 1)
    pos = s * Qm1
    lo = np.floor(pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, Q - 1)
    w_hi = (pos - lo).astype(np.float32)
    w_lo = 1.0 - w_hi

    g_lo = g[np.arange(N)[:, None], np.arange(k)[None, :], lo[:, None]]
    g_hi = g[np.arange(N)[:, None], np.arange(k)[None, :], hi[:, None]]
    g_obs = (w_lo[:, None] * g_lo + w_hi[:, None] * g_hi).astype(np.float32)

    logits = cfg.kappa * (y_t * g_obs)
    probs = _softmax_np(logits, axis=1).astype(np.float32)
    t = np.array([rng.choice(k, p=probs[i]) for i in range(N)], dtype=np.int64)

    y_full_t = y_full[np.arange(N), t, :]
    y = (w_lo * y_full_t[np.arange(N), lo] + w_hi * y_full_t[np.arange(N), hi]).astype(np.float32)

    return {
        "x": x.astype(np.float32),
        "z": z.astype(np.float32),
        "t": t,
        "s": s.astype(np.float32),
        "y": y.astype(np.float32),
        "y_full": y_full.astype(np.float32),
        "s_grid": s_grid.astype(np.float32),
    }


# ============================================================
#  Dataset / splits / scaling
# ============================================================

class NewsTwinDataset(Dataset):
    """
    If y_full is provided, __getitem__ returns:
      (xz, t, s, y_norm, y_full_row)
    where y_full_row: (k,Q) in ORIGINAL scale.
    """
    def __init__(self, xz, t, s, y_norm, y_full: Optional[np.ndarray] = None):
        self.xz = torch.from_numpy(xz).float()
        self.t = torch.from_numpy(t).long()
        self.s = torch.from_numpy(s).float()
        self.y = torch.from_numpy(y_norm).float()
        self.y_full = None if y_full is None else torch.from_numpy(y_full).float()

    def __len__(self):
        return self.xz.shape[0]

    def __getitem__(self, idx):
        if self.y_full is None:
            return self.xz[idx], self.t[idx], self.s[idx], self.y[idx]
        return self.xz[idx], self.t[idx], self.s[idx], self.y[idx], self.y_full[idx]

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
    xz_mean: np.ndarray
    xz_std: np.ndarray
    y_mean: float
    y_std: float

def fit_scalers(xz_train: np.ndarray, y_train: np.ndarray) -> Scalers:
    xz_mean = xz_train.mean(axis=0)
    xz_std = xz_train.std(axis=0)
    xz_std = np.where(xz_std < 1e-6, 1.0, xz_std)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-6 else 1.0)
    return Scalers(xz_mean=xz_mean, xz_std=xz_std, y_mean=y_mean, y_std=y_std)

def apply_xz_scaler(xz: np.ndarray, sc: Scalers) -> np.ndarray:
    return (xz - sc.xz_mean) / sc.xz_std

def apply_y_scaler(y: np.ndarray, sc: Scalers) -> np.ndarray:
    return (y - sc.y_mean) / sc.y_std

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
    def __init__(self, xz_dim: int, u_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = MLP(xz_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def forward(self, xz):
        h = self.backbone(xz)
        lv = self.logvar(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class PriorV(nn.Module):
    def __init__(self, xz_dim: int, k: int, v_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_emb = nn.Embedding(k, 32)
        self.backbone = MLP(xz_dim + 32, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, v_dim)
        self.logvar = nn.Linear(hidden, v_dim)

    def forward(self, xz, t):
        te = self.t_emb(t.long())
        h = self.backbone(torch.cat([xz, te], dim=-1))
        lv = self.logvar(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class EncoderUV(nn.Module):
    def __init__(self, xz_dim: int, k: int, u_dim: int, v_dim: int, dose_rbf: Optional["DoseRBF"],
                 hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_emb = nn.Embedding(k, 32)
        self.dose_rbf = dose_rbf
        rbf_dim = int(dose_rbf.dim) if dose_rbf is not None else 0
        in_dim = xz_dim + 32 + 1 + rbf_dim + 1  # xz + t + s + rbf(s) + y
        self.backbone = MLP(in_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.u_mu = nn.Linear(hidden, u_dim)
        self.u_lv = nn.Linear(hidden, u_dim)
        self.v_mu = nn.Linear(hidden, v_dim)
        self.v_lv = nn.Linear(hidden, v_dim)

    def forward(self, xz, t, s, y_norm):
        te = self.t_emb(t.long())
        feats = [xz, te, s.unsqueeze(-1)]
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
    def __init__(self, xz_dim: int, k: int, u_dim: int, v_dim: int, dose_rbf: Optional["DoseRBF"],
                 hidden: int = 512, depth: int = 2, dropout: float = 0.1,
                 y_logvar_min: float = -2.5, y_logvar_max: float = 4.0):
        super().__init__()
        self.k = int(k)
        self.t_emb = nn.Embedding(k, 32)
        self.dose_rbf = dose_rbf
        self.y_logvar_min = float(y_logvar_min)
        self.y_logvar_max = float(y_logvar_max)

        rbf_dim = int(dose_rbf.dim) if dose_rbf is not None else 0
        self.trunk = MLP(xz_dim + u_dim, hidden, hidden, depth=depth, dropout=dropout)
        head_in = hidden + v_dim + 32 + 1 + rbf_dim

        self.head_mu = MLP(head_in, hidden, 1, depth=2, dropout=dropout)
        self.head_lv = MLP(head_in, hidden, 1, depth=2, dropout=dropout)

    def forward(self, xz, t, s, u, v):
        te = self.t_emb(t.long())
        h = self.trunk(torch.cat([xz, u], dim=-1))
        feats = [h, v, te, s.unsqueeze(-1)]
        if self.dose_rbf is not None:
            feats.append(self.dose_rbf(s))
        hh = torch.cat(feats, dim=-1)
        mu = self.head_mu(hh).squeeze(-1)
        lv = self.head_lv(hh).squeeze(-1).clamp(self.y_logvar_min, self.y_logvar_max)
        return mu, lv


# ============================================================
#  Sampler helpers
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


# ============================================================
#  Support mask (positivity restriction)
# ============================================================

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
        mask_tt = (s_grid >= qlo[tt]) & (s_grid <= qhi[tt])
        mask[tt] = mask_tt

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
#  Evaluation helpers (MISE / DPE / PE)
# ============================================================

def _apply_support_mask_to_surface(y_hat: np.ndarray, support_mask_tq: Optional[np.ndarray], fill_value: float = -1e9) -> np.ndarray:
    if support_mask_tq is None:
        return y_hat
    mask = np.asarray(support_mask_tq, dtype=bool)
    return np.where(mask[None, :, :], y_hat, float(fill_value))

@torch.no_grad()
def evaluate_curves(
    prior_u: "PriorU",
    prior_v: "PriorV",
    enc: "EncoderUV",
    dec: "DecoderUV",
    loader: DataLoader,
    y_full: np.ndarray,
    s_grid: np.ndarray,
    idx_subset: np.ndarray,
    scalers: "Scalers",
    device: str,
    u_source: str = "posterior",
    support_mask_tq: Optional[np.ndarray] = None,
    mc_eval_u: int = 0,
    sample_v_eval: bool = True,
    dec_chunk: int = 8192,
) -> Dict[str, Any]:
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
            xz, t, s, y_norm = batch
        else:
            xz, t, s, y_norm, _yfull_row = batch

        B = int(xz.shape[0])
        cur_ids = idx_subset[offset:offset + B]
        offset += B

        xz = xz.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        if u_source == "posterior":
            mu_u, lv_u, _, _ = enc(xz, t, s, y_norm)
        else:
            mu_u, lv_u = prior_u(xz)

        M = int(mc_eval_u) if (mc_eval_u and mc_eval_u > 0) else 1

        y_sum = np.zeros((B, k, Q), dtype=np.float64)
        y_sum2 = np.zeros((B, k, Q), dtype=np.float64)

        q_chunk = max(1, int(dec_chunk // max(1, B)))
        q_chunk = min(q_chunk, Q)

        for _m in range(M):
            u_samp = reparam(mu_u, lv_u) if M > 1 else mu_u

            for tt in range(k):
                t_tt = torch.full((B,), tt, device=device, dtype=torch.long)
                mu_v_tt, lv_v_tt = prior_v(xz, t_tt)
                v_tt = reparam(mu_v_tt, lv_v_tt) if (sample_v_eval and M > 1) else mu_v_tt

                for q0 in range(0, Q, q_chunk):
                    q1 = min(Q, q0 + q_chunk)
                    qc = q1 - q0

                    xz_rep = xz.repeat_interleave(qc, dim=0)
                    u_rep  = u_samp.repeat_interleave(qc, dim=0)
                    v_rep  = v_tt.repeat_interleave(qc, dim=0)

                    t_rep = torch.full((B * qc,), tt, device=device, dtype=torch.long)
                    s_rep = s_grid_t[q0:q1].repeat(B)

                    mu_hat_norm, lv_hat_norm = dec(xz_rep, t_rep, s_rep, u_rep, v_rep)

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

        y_plan = _apply_support_mask_to_surface(y_mean, support_mask_tq)

        gt_best_dose = gt.max(axis=2)
        gt_best_overall = gt_best_dose.max(axis=1)

        shat_per_t = y_plan.argmax(axis=2).astype(np.int64)
        gt_at_shat = np.take_along_axis(gt, shat_per_t[:, :, None], axis=2).squeeze(2)
        dpe_sum += float(np.sum(gt_best_dose - gt_at_shat))

        flat = y_plan.reshape(B, -1).argmax(axis=1).astype(np.int64)
        that = (flat // Q).astype(np.int64)
        qhat = (flat % Q).astype(np.int64)
        gt_at_policy = gt[np.arange(B), that, qhat]
        pe_sum += float(np.sum(gt_best_overall - gt_at_policy))

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



# z-scores for central Gaussian intervals
# 50% -> 0.67449, 80% -> 1.28155, 90% -> 1.64485, 95% -> 1.95996
_Z_TABLE = {
    0.50: 0.6744897501960817,
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}

@torch.no_grad()
def evaluate_predictive_interval_coverage(
    prior_u: "PriorU",
    prior_v: "PriorV",
    enc: "EncoderUV",
    dec: "DecoderUV",
    loader: DataLoader,
    scalers: "Scalers",
    device: str,
    levels: List[float],
    mc: int = 50,
    u_source: str = "prior",          # recommended for "predictive" (planning-time) uncertainty
    use_posterior_v: bool = False,    # keep False for "predictive"; True uses y in encoder for v
    sample_v: bool = True,
    eps_std: float = 1e-8,
) -> Dict[str, Any]:
    """
    Computes empirical coverage on factual test points for central intervals:
      [mu - z*std, mu + z*std], where std includes epistemic (MC over u,v) + aleatoric (decoder var).

    We report coverage + average interval width per level. This is a descriptive diagnostic,
    not a calibration claim.
    """
    prior_u.eval(); prior_v.eval(); enc.eval(); dec.eval()

    # normalize and unique-sort levels
    lvls = [float(x) for x in levels]
    lvls = sorted(set(lvls))
    for L in lvls:
        if L <= 0 or L >= 1:
            raise ValueError(f"Coverage level must be in (0,1); got {L}")
        if L not in _Z_TABLE:
            # allow arbitrary L via normal approximation? keep simple: require known levels.
            raise ValueError(f"Level {L} not in built-in z-table. Use one of {sorted(_Z_TABLE.keys())}.")

    cov_hits = {L: 0 for L in lvls}
    width_sum = {L: 0.0 for L in lvls}
    n_total = 0

    M = max(1, int(mc))

    for batch in loader:
        if len(batch) == 4:
            xz, t, s, y_norm = batch
        else:
            xz, t, s, y_norm, _ = batch

        xz = xz.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        B = int(xz.shape[0])

        # choose u source
        if u_source == "posterior":
            mu_u_q, lv_u_q, mu_v_q, lv_v_q = enc(xz, t, s, y_norm)
        else:
            mu_u_q, lv_u_q = prior_u(xz)
            mu_v_q, lv_v_q = None, None

        # choose v source (default: prior, to avoid conditioning on y)
        if use_posterior_v and (mu_v_q is not None):
            mu_v0, lv_v0 = mu_v_q, lv_v_q
        else:
            mu_v0, lv_v0 = prior_v(xz, t)

        sum_mu = torch.zeros((B,), device=device)
        sum_second = torch.zeros((B,), device=device)

        for _ in range(M):
            u_s = reparam(mu_u_q, lv_u_q) if M > 1 else mu_u_q
            if sample_v and M > 1:
                v_s = reparam(mu_v0, lv_v0)
            else:
                v_s = mu_v0

            mu_hat_norm, lv_hat_norm = dec(xz, t, s, u_s, v_s)
            var_dec_norm = torch.exp(lv_hat_norm).clamp_min(0.0)

            sum_mu += mu_hat_norm
            sum_second += (mu_hat_norm * mu_hat_norm + var_dec_norm)

        mean_norm = sum_mu / float(M)
        second_norm = sum_second / float(M)
        var_norm = (second_norm - mean_norm * mean_norm).clamp_min(0.0)
        std_norm = torch.sqrt(var_norm + eps_std)

        mean = mean_norm * scalers.y_std + scalers.y_mean
        std = std_norm * scalers.y_std
        y_true = y_norm * scalers.y_std + scalers.y_mean

        for L in lvls:
            z = float(_Z_TABLE[L])
            lo = mean - z * std
            hi = mean + z * std
            hit = ((y_true >= lo) & (y_true <= hi)).sum().item()
            cov_hits[L] += int(hit)
            width_sum[L] += float((2.0 * z * std).sum().item())

        n_total += B

    rows = []
    for L in lvls:
        rows.append({
            "level": float(L),
            "z": float(_Z_TABLE[L]),
            "coverage": float(cov_hits[L] / max(1, n_total)),
            "avg_width": float(width_sum[L] / max(1, n_total)),
            "n": int(n_total),
        })

    return {
        "u_source": str(u_source),
        "mc": int(M),
        "use_posterior_v": bool(use_posterior_v),
        "sample_v": bool(sample_v),
        "rows": rows,
    }

def plot_interval_coverage(rows: List[Dict[str, Any]], outpath: str, title: str):
    xs = [r["level"] for r in rows]
    ys = [r["coverage"] for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o", label="Empirical coverage")
    plt.plot(xs, xs, linestyle="--", label="Ideal (y=x)")
    plt.xlabel("Nominal interval level")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()



@torch.no_grad()
def evaluate_oracle_overlap_topk(
    prior_u: "PriorU",
    prior_v: "PriorV",
    enc: "EncoderUV",
    dec: "DecoderUV",
    loader: DataLoader,
    y_full: np.ndarray,          # (N,k,Q) oracle outcomes in original scale
    s_grid: np.ndarray,
    idx_subset: np.ndarray,      # mapping dataset-order -> global index
    scalers: "Scalers",
    device: str,
    ks: List[int],
    u_source: str = "prior",
    support_mask_tq: Optional[np.ndarray] = None,
    mc: int = 20,
    sample_v: bool = True,
    dec_chunk: int = 8192,
) -> Dict[str, Any]:
    """
    Top-K overlap between predicted planner (based on predicted mean surface) and oracle optimum.
    We report:
      - joint (t,q) top-K accuracy
      - treatment top-K accuracy (rank treatments by max_q y_hat(t,q))
      - dose top-K accuracy conditional on oracle treatment t* (rank q by y_hat(t*,q))
    Oracle is computed under the same support mask if provided.
    """
    prior_u.eval(); prior_v.eval(); enc.eval(); dec.eval()

    k = int(y_full.shape[1])
    Q = int(y_full.shape[2])
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    ks = sorted(set(int(x) for x in ks))
    if any(x <= 0 for x in ks):
        raise ValueError(f"All ks must be positive; got {ks}")
    kmax = max(ks)

    hits_joint = {K: 0 for K in ks}
    hits_treat = {K: 0 for K in ks}
    hits_dose  = {K: 0 for K in ks}

    rank_joint_sum = 0.0
    rank_treat_sum = 0.0
    rank_dose_sum  = 0.0

    n_total = 0
    M = max(1, int(mc))

    offset = 0
    for batch in loader:
        if len(batch) == 4:
            xz, t_obs, s_obs, y_norm = batch
        else:
            xz, t_obs, s_obs, y_norm, _ = batch

        B = int(xz.shape[0])
        cur_ids = idx_subset[offset:offset + B]
        offset += B

        xz = xz.to(device, non_blocking=True)
        t_obs = t_obs.to(device, non_blocking=True)
        s_obs = s_obs.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)

        # choose u source
        if u_source == "posterior":
            mu_u, lv_u, _, _ = enc(xz, t_obs, s_obs, y_norm)
        else:
            mu_u, lv_u = prior_u(xz)

        # predicted mean surface y_hat: (B,k,Q)
        y_sum = np.zeros((B, k, Q), dtype=np.float64)

        q_chunk = max(1, int(dec_chunk // max(1, B)))
        q_chunk = min(q_chunk, Q)

        for _ in range(M):
            u_s = reparam(mu_u, lv_u) if M > 1 else mu_u

            for tt in range(k):
                t_tt = torch.full((B,), tt, device=device, dtype=torch.long)
                mu_v_tt, lv_v_tt = prior_v(xz, t_tt)
                if sample_v and M > 1:
                    v_tt = reparam(mu_v_tt, lv_v_tt)
                else:
                    v_tt = mu_v_tt

                for q0 in range(0, Q, q_chunk):
                    q1 = min(Q, q0 + q_chunk)
                    qc = q1 - q0

                    xz_rep = xz.repeat_interleave(qc, dim=0)
                    u_rep  = u_s.repeat_interleave(qc, dim=0)
                    v_rep  = v_tt.repeat_interleave(qc, dim=0)

                    t_rep = torch.full((B * qc,), tt, device=device, dtype=torch.long)
                    s_rep = s_grid_t[q0:q1].repeat(B)

                    mu_hat_norm, _lv_hat_norm = dec(xz_rep, t_rep, s_rep, u_rep, v_rep)
                    mu_hat = (mu_hat_norm * scalers.y_std + scalers.y_mean).view(B, qc)
                    y_sum[:, tt, q0:q1] += mu_hat.detach().cpu().numpy().astype(np.float64)

        y_mean = (y_sum / float(M)).astype(np.float64)

        # apply mask to predictions (feasible planning)
        y_plan = _apply_support_mask_to_surface(y_mean, support_mask_tq, fill_value=-1e18)

        # oracle under same support mask
        gt = y_full[cur_ids].astype(np.float64)  # (B,k,Q)
        gt_plan = _apply_support_mask_to_surface(gt, support_mask_tq, fill_value=-1e18)

        # oracle joint
        gt_flat = gt_plan.reshape(B, -1)
        oracle_flat = gt_flat.argmax(axis=1).astype(np.int64)          # in [0, k*Q)
        oracle_t = (oracle_flat // Q).astype(np.int64)
        oracle_q = (oracle_flat % Q).astype(np.int64)

        # predicted joint ranking
        pred_flat = y_plan.reshape(B, -1)
        topk_pred_flat = np.argpartition(-pred_flat, kth=min(kmax-1, pred_flat.shape[1]-1), axis=1)[:, :kmax]
        # sort the top-k by score (descending)
        row_scores = np.take_along_axis(pred_flat, topk_pred_flat, axis=1)
        order = np.argsort(-row_scores, axis=1)
        topk_pred_flat = np.take_along_axis(topk_pred_flat, order, axis=1)

        # compute joint rank (1-indexed) of oracle in full list (approx via argsort)
        # for reporting, do full sort only once per batch (still cheap: k*Q ~ 1k)
        full_order_flat = np.argsort(-pred_flat, axis=1)
        inv_rank_flat = np.empty_like(full_order_flat)
        inv_rank_flat[np.arange(B)[:, None], full_order_flat] = np.arange(full_order_flat.shape[1])[None, :]
        rank_joint_sum += float((inv_rank_flat[np.arange(B), oracle_flat] + 1).sum())

        for K in ks:
            hit = (topk_pred_flat[:, :K] == oracle_flat[:, None]).any(axis=1).sum()
            hits_joint[K] += int(hit)

        # treatment ranking by best dose
        pred_best_t = y_plan.max(axis=2)  # (B,k)
        topk_pred_t = np.argpartition(-pred_best_t, kth=min(kmax-1, k-1), axis=1)[:, :kmax]
        t_scores = np.take_along_axis(pred_best_t, topk_pred_t, axis=1)
        t_order = np.argsort(-t_scores, axis=1)
        topk_pred_t = np.take_along_axis(topk_pred_t, t_order, axis=1)

        full_order_t = np.argsort(-pred_best_t, axis=1)
        inv_rank_t = np.empty_like(full_order_t)
        inv_rank_t[np.arange(B)[:, None], full_order_t] = np.arange(k)[None, :]
        rank_treat_sum += float((inv_rank_t[np.arange(B), oracle_t] + 1).sum())

        for K in ks:
            hit = (topk_pred_t[:, :K] == oracle_t[:, None]).any(axis=1).sum()
            hits_treat[K] += int(hit)

        # dose ranking conditional on oracle treatment
        pred_dose_oracle_t = y_plan[np.arange(B), oracle_t, :]  # (B,Q)
        topk_pred_q = np.argpartition(-pred_dose_oracle_t, kth=min(kmax-1, Q-1), axis=1)[:, :kmax]
        q_scores = np.take_along_axis(pred_dose_oracle_t, topk_pred_q, axis=1)
        q_order = np.argsort(-q_scores, axis=1)
        topk_pred_q = np.take_along_axis(topk_pred_q, q_order, axis=1)

        full_order_q = np.argsort(-pred_dose_oracle_t, axis=1)
        inv_rank_q = np.empty_like(full_order_q)
        inv_rank_q[np.arange(B)[:, None], full_order_q] = np.arange(Q)[None, :]
        rank_dose_sum += float((inv_rank_q[np.arange(B), oracle_q] + 1).sum())

        for K in ks:
            hit = (topk_pred_q[:, :K] == oracle_q[:, None]).any(axis=1).sum()
            hits_dose[K] += int(hit)

        n_total += B

    rows = []
    for K in ks:
        rows.append({
            "K": int(K),
            "joint_topk_acc": float(hits_joint[K] / max(1, n_total)),
            "treat_topk_acc": float(hits_treat[K] / max(1, n_total)),
            "dose_topk_acc_cond_oracle_t": float(hits_dose[K] / max(1, n_total)),
        })

    out = {
        "u_source": str(u_source),
        "mc": int(M),
        "sample_v": bool(sample_v),
        "n": int(n_total),
        "mean_rank_joint": float(rank_joint_sum / max(1, n_total)),
        "mean_rank_treat": float(rank_treat_sum / max(1, n_total)),
        "mean_rank_dose": float(rank_dose_sum / max(1, n_total)),
        "rows": rows,
    }
    return out

def plot_oracle_overlap_topk(rows: List[Dict[str, Any]], outpath: str, title: str):
    xs = [r["K"] for r in rows]
    yj = [r["joint_topk_acc"] for r in rows]
    yt = [r["treat_topk_acc"] for r in rows]
    yq = [r["dose_topk_acc_cond_oracle_t"] for r in rows]
    plt.figure()
    plt.plot(xs, yj, marker="o", label="Joint (t,q)")
    plt.plot(xs, yt, marker="o", label="Treatment t")
    plt.plot(xs, yq, marker="o", label="Dose q | oracle t")
    plt.xlabel("K")
    plt.ylabel("Top-K accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
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
    xz_dim: int,
    cfg: TrainConfig,
    dose_rbf: Optional[DoseRBF],
    eval_u: str = "both",
    support_mask_tq: Optional[np.ndarray] = None,
):
    device = cfg.device
    Q = int(y_full.shape[2])
    s_grid_t = torch.from_numpy(s_grid).float().to(device)

    prior_u = PriorU(xz_dim=xz_dim, u_dim=cfg.u_dim, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    prior_v = PriorV(xz_dim=xz_dim, k=k, v_dim=cfg.v_dim, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    enc = EncoderUV(xz_dim=xz_dim, k=k, u_dim=cfg.u_dim, v_dim=cfg.v_dim, dose_rbf=dose_rbf,
                    hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    dec = DecoderUV(xz_dim=xz_dim, k=k, u_dim=cfg.u_dim, v_dim=cfg.v_dim, dose_rbf=dose_rbf,
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
            )
        return outs

    for epoch in range(1, cfg.epochs + 1):
        prior_u.train(); prior_v.train(); enc.train(); dec.train()

        beta = cfg.beta_kl * min(1.0, float(epoch) / float(cfg.kl_anneal_epochs)) if cfg.kl_anneal_epochs > 0 else cfg.beta_kl

        lam_prior = _lin_ramp(epoch, cfg.prior_recon_anneal_epochs, cfg.lambda_prior_recon)
        lam_bal   = _lin_ramp(epoch, cfg.balance_anneal_epochs,    cfg.lambda_balance)
        lam_surf  = _lin_ramp(epoch, cfg.surface_anneal_epochs,    cfg.lambda_surface)

        loss_sum = recon_sum = recon_prior_sum = klu_raw_sum = klv_raw_sum = klfb_sum = lat_sum = 0.0
        surf_sum = bal_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                xz, t, s, y_norm = batch
                y_full_row = None
            else:
                xz, t, s, y_norm, y_full_row = batch

            xz = xz.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            if y_full_row is not None:
                y_full_row = y_full_row.to(device, non_blocking=True)

            mu_u_q, lv_u_q, mu_v_q, lv_v_q = enc(xz, t, s, y_norm)
            u = reparam(mu_u_q, lv_u_q)
            v = reparam(mu_v_q, lv_v_q)

            mu_u_p, lv_u_p = prior_u(xz)
            mu_v_p, lv_v_p = prior_v(xz, t)

            mu_hat, lv_hat = dec(xz, t, s, u, v)
            recon = gaussian_nll(y_norm, mu_hat, lv_hat)

            mu_hat_p, lv_hat_p = dec(xz, t, s, mu_u_p, mu_v_p)
            recon_prior = gaussian_nll(y_norm, mu_hat_p, lv_hat_p)

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

            lat = torch.tensor(0.0, device=device)
            if cfg.lambda_lat > 0 and cfg.n_cf > 0:
                B = xz.shape[0]
                u_anchor = mu_u_q.detach()
                for _ in range(int(cfg.n_cf)):
                    t_cf = torch.randint(low=0, high=k, size=(B,), device=device)
                    s_cf = torch.rand((B,), device=device)

                    mu_v_cf, lv_v_cf = prior_v(xz, t_cf)
                    v_cf = reparam(mu_v_cf, lv_v_cf)

                    mu_cf, _lv_cf = dec(xz, t_cf, s_cf, u, v_cf)
                    y_cf = mu_cf
                    mu_u_cf, _, _, _ = enc(xz, t_cf, s_cf, y_cf)
                    lat = lat + F.mse_loss(mu_u_cf, u_anchor)
                lat = lat / float(cfg.n_cf)

            bal = torch.tensor(0.0, device=device)
            if lam_bal > 0:
                u_for_bal = mu_u_q if cfg.balance_on == "mu" else u
                if cfg.balance_kind == "hsic":
                    bal = hsic_rbf(u_for_bal, t, k=k, sigma_u=float(cfg.balance_sigma), sigma_t=-1.0)
                else:
                    bal = mmd_pairwise_groups(u_for_bal, t, k=k, sigma=float(cfg.balance_sigma))

            surf = torch.tensor(0.0, device=device)
            if (lam_surf > 0) and (y_full_row is not None) and (cfg.n_surface > 0):
                B = xz.shape[0]
                n_surf = int(cfg.n_surface)

                t_surf = torch.randint(low=0, high=k, size=(B, n_surf), device=device)
                q_surf = torch.randint(low=0, high=Q, size=(B, n_surf), device=device)

                s_surf = s_grid_t[q_surf]  # (B,n_surf)

                y_target = y_full_row.gather(1, t_surf.unsqueeze(-1).expand(B, n_surf, Q)).gather(
                    2, q_surf.unsqueeze(-1)
                ).squeeze(-1)
                y_target_norm = (y_target - float(scalers.y_mean)) / float(scalers.y_std)

                u_s = mu_u_q if cfg.surface_u_source == "posterior_mu" else mu_u_p

                t_flat = t_surf.reshape(-1)
                s_flat = s_surf.reshape(-1)
                xz_rep = xz.repeat_interleave(n_surf, dim=0)
                u_rep  = u_s.repeat_interleave(n_surf, dim=0)

                mu_v_s, _lv_v_s = prior_v(xz_rep, t_flat)
                v_rep = mu_v_s

                mu_surf, lv_surf = dec(xz_rep, t_flat, s_flat, u_rep, v_rep)
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
#  Main utils
# ============================================================

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _parse_float_list(s: str) -> List[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out

def _parse_int_list(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out

def _write_csv_rows(rows: List[Dict[str, Any]], outpath: str, field_order: Optional[List[str]] = None):
    if not rows:
        return
    if field_order is None:
        field_order = list(rows[0].keys())
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(",".join(field_order) + "\n")
        for r in rows:
            vals = []
            for k in field_order:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--subsample_n", type=int, default=5000)

    p.add_argument("--k", type=int, default=8)
    p.add_argument("--Q", type=int, default=65)
    p.add_argument("--kappa", type=float, default=7.0)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1000)

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

    p.add_argument("--lambda_balance", type=float, default=0.3)
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
    p.add_argument("--outdir", type=str, default="./out")

    # NEW: coverage
    p.add_argument("--report_coverage", action="store_true")
    p.add_argument("--coverage_levels", type=str, default="0.5,0.8,0.9")
    p.add_argument("--coverage_u_source", type=str, default="prior", choices=["prior", "posterior"])
    p.add_argument("--coverage_mc", type=int, default=50)
    p.add_argument("--coverage_use_posterior_v", action="store_true")
    p.add_argument("--coverage_sample_v", action="store_true")

    # NEW: oracle overlap / top-k
    p.add_argument("--report_oracle_overlap", action="store_true")
    p.add_argument("--overlap_ks", type=str, default="1,3,5,10")
    p.add_argument("--overlap_u_source", type=str, default="prior", choices=["prior", "posterior"])
    p.add_argument("--overlap_mc", type=int, default=20)
    p.add_argument("--overlap_sample_v", action="store_true")

    args = p.parse_args()
    seed_everything(args.seed)
    _ensure_dir(args.outdir)

    x_all, z_all = load_news_xz_drnet(args.data_dir)
    N_all = int(x_all.shape[0])
    n = int(args.subsample_n)
    if n <= 0 or n > N_all:
        raise ValueError(f"--subsample_n must be in [1, {N_all}]")
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(N_all, size=n, replace=False)
    x = x_all[idx]
    z = z_all[idx]
    print(f"[Load] x={x_all.shape} z={z_all.shape} | subsample_n={n}")

    bcfg = NewsBenchmarkConfig(k=args.k, Q=args.Q, kappa=args.kappa, beta=args.beta, seed=args.seed)
    bench = build_news_semi_synthetic(x=x, z=z, cfg=bcfg)
    t = bench["t"]; s = bench["s"]; y = bench["y"]; y_full = bench["y_full"]; s_grid = bench["s_grid"]
    print(f"[Bench] y_full={y_full.shape} | y range [{y.min():.3f},{y.max():.3f}]")

    dose_rbf: Optional[DoseRBF] = None
    if not args.no_dose_rbf:
        sigma = None if args.dose_rbf_sigma < 0 else float(args.dose_rbf_sigma)
        dose_rbf = DoseRBF(dim=int(args.dose_rbf_dim), sigma=sigma)

    N = int(y_full.shape[0])
    idx_train, idx_val, idx_test = split_indices(N, seed=args.seed, frac_train=0.6, frac_val=0.2)

    x_log = np.log1p(bench["x"])
    xz = np.concatenate([x_log, bench["z"]], axis=1)
    scalers = fit_scalers(xz[idx_train], y[idx_train])
    xz_std = apply_xz_scaler(xz, scalers)
    y_norm = apply_y_scaler(y, scalers)

    ds_train = NewsTwinDataset(xz_std[idx_train], t[idx_train], s[idx_train], y_norm[idx_train], y_full=y_full[idx_train])
    ds_val   = NewsTwinDataset(xz_std[idx_val],   t[idx_val],   s[idx_val],   y_norm[idx_val],   y_full=None)
    ds_test  = NewsTwinDataset(xz_std[idx_test],  t[idx_test],  s[idx_test],  y_norm[idx_test],  y_full=None)

    support_mask_tq = None
    if not args.no_support_mask:
        support_mask_tq, support_info = build_support_mask(
            t_train=t[idx_train],
            s_train=s[idx_train],
            s_grid=s_grid,
            k=args.k,
            Q=args.Q,
            q_low=float(args.support_q_low),
            q_high=float(args.support_q_high),
            min_count=int(args.support_min_count),
        )
        print(f"[SupportMask] enabled | allowed_frac={support_info['allowed_frac']:.3f}")
    else:
        print("[SupportMask] disabled")

    if args.use_sampler:
        w_np = compute_sampler_weights(
            t=t[idx_train], s=s[idx_train], k=args.k, Q=args.Q,
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
        k=args.k,
        xz_dim=xz_std.shape[1],
        cfg=cfg,
        dose_rbf=dose_rbf,
        eval_u=args.eval_u,
        support_mask_tq=support_mask_tq,
    )

    # Standard test eval
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
        )
        msg = f"[Test-{usrc}] MISE={o['MISE']:.6f} DPE={o['DPE']:.6f} PE={o['PE']:.6f} (mc_u={o.get('mc_eval_u',0)} sample_v={o.get('sample_v_eval',False)})"
        if "MISE_in" in o:
            msg += f" | MISE(in)={o['MISE_in']:.6f} MISE(out)={o['MISE_out']:.6f}"
        print(msg)

    # NEW: predictive interval coverage
    if args.report_coverage:
        levels = _parse_float_list(args.coverage_levels)
        cov = evaluate_predictive_interval_coverage(
            prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            scalers=scalers,
            device=args.device,
            levels=levels,
            mc=int(args.coverage_mc),
            u_source=str(args.coverage_u_source),
            use_posterior_v=bool(args.coverage_use_posterior_v),
            sample_v=bool(args.coverage_sample_v),
        )
        print(f"[Coverage-{cov['u_source']}] mc={cov['mc']} use_post_v={cov['use_posterior_v']} sample_v={cov['sample_v']}")
        for r in cov["rows"]:
            print(f"  level={r['level']:.2f} cov={r['coverage']:.4f} avg_width={r['avg_width']:.4f} (z={r['z']:.3f}, n={r['n']})")

        csv_path = join(args.outdir, "predictive_interval_coverage.csv")
        _write_csv_rows(cov["rows"], csv_path, field_order=["level", "z", "coverage", "avg_width", "n"])

        plot_path = join(args.outdir, "predictive_interval_coverage.png")
        plot_interval_coverage(
            cov["rows"],
            plot_path,
            title=f"Predictive interval coverage (u={cov['u_source']}, mc={cov['mc']})"
        )
        print(f"[Coverage] Saved: {csv_path}")
        print(f"[Coverage] Saved: {plot_path}")

    # NEW: oracle overlap / top-k
    if args.report_oracle_overlap:
        ks = _parse_int_list(args.overlap_ks)
        overlap = evaluate_oracle_overlap_topk(
            prior_u=prior_u, prior_v=prior_v, enc=enc, dec=dec,
            loader=test_loader,
            y_full=y_full,
            s_grid=s_grid,
            idx_subset=idx_test,
            scalers=scalers,
            device=args.device,
            ks=ks,
            u_source=str(args.overlap_u_source),
            support_mask_tq=support_mask_tq,
            mc=int(args.overlap_mc),
            sample_v=bool(args.overlap_sample_v),
            dec_chunk=int(args.eval_dec_chunk),
        )
        print(f"[OracleOverlap-{overlap['u_source']}] mc={overlap['mc']} sample_v={overlap['sample_v']} n={overlap['n']}")
        print(f"  mean_rank_joint={overlap['mean_rank_joint']:.2f} mean_rank_treat={overlap['mean_rank_treat']:.2f} mean_rank_dose={overlap['mean_rank_dose']:.2f}")
        for r in overlap["rows"]:
            print(f"  K={r['K']:<3d} joint={r['joint_topk_acc']:.4f} treat={r['treat_topk_acc']:.4f} dose|t*={r['dose_topk_acc_cond_oracle_t']:.4f}")

        csv_path = join(args.outdir, "oracle_overlap_topk.csv")
        _write_csv_rows(overlap["rows"], csv_path, field_order=["K", "joint_topk_acc", "treat_topk_acc", "dose_topk_acc_cond_oracle_t"])

        plot_path = join(args.outdir, "oracle_overlap_topk.png")
        plot_oracle_overlap_topk(
            overlap["rows"],
            plot_path,
            title=f"Oracle overlap / Top-K (u={overlap['u_source']}, mc={overlap['mc']})"
        )
        print(f"[OracleOverlap] Saved: {csv_path}")
        print(f"[OracleOverlap] Saved: {plot_path}")


if __name__ == "__main__":
    main()
