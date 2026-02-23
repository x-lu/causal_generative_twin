#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scanpy as sc
import scipy.sparse
from sklearn.metrics import r2_score


# ============================================================
#  Seed / schema / helpers
# ============================================================

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def inspect_adata_schema(adata):
    print("=== AnnData schema ===")
    print(f"n_obs={adata.n_obs} n_vars={adata.n_vars} X_type={type(adata.X)}")
    print("obs.columns:", list(adata.obs.columns))
    if "fields" in adata.uns:
        try:
            print("uns['fields']:", dict(adata.uns["fields"]))
        except Exception:
            print("uns['fields']: <present but not dict-castable>")
    else:
        print("uns['fields']: <missing>")
    print("======================")

def resolve_field(adata, key: Optional[str], default: Optional[str] = None):
    """
    If key is in adata.uns['fields'], return mapped value.
    Else return key itself. If key is None, use default.
    """
    if key is None:
        key = default
    if key is None:
        return None
    if "fields" in adata.uns and key in adata.uns["fields"]:
        return adata.uns["fields"][key]
    return key

def as_list_of_str(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (tuple, set)):
        x = list(x)
    if isinstance(x, str):
        return [x]
    return [str(k) for k in x]

def parse_combo_string(s: str) -> List[str]:
    s = str(s)
    return [p.strip() for p in s.split("+") if p.strip() != ""]

def parse_dose_combo(d) -> List[float]:
    if isinstance(d, (float, int, np.floating, np.integer)):
        return [float(d)]
    parts = parse_combo_string(str(d))
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            out.append(1.0)
    return out

def safe_to_dense_rows(X, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        D = int(X.shape[1])
        return np.zeros((0, D), dtype=np.float32)
    if scipy.sparse.issparse(X):
        return X[rows].toarray().astype(np.float32)
    return np.asarray(X[rows], dtype=np.float32)

def safe_to_dense_row(X, i: int) -> np.ndarray:
    if scipy.sparse.issparse(X):
        return X.getrow(i).toarray().astype(np.float32).reshape(-1)
    return np.asarray(X[i], dtype=np.float32).reshape(-1)

def one_hot_from_categories(values: np.ndarray) -> Tuple[torch.Tensor, Dict[str, int]]:
    vals = np.asarray(values).astype(str)
    uniq = np.unique(vals)
    mapping = {u: i for i, u in enumerate(uniq.tolist())}
    idx = np.array([mapping[v] for v in vals], dtype=np.int64)
    ohe = torch.eye(len(uniq), dtype=torch.float32)[torch.from_numpy(idx)]
    return ohe, mapping

def _to_int01(v) -> int:
    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))
    if isinstance(v, (int, np.integer)):
        return 1 if int(v) != 0 else 0
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


# ============================================================
#  Lazy genes accessor (Fix 1: support 2D indexing genes[idx, :])
# ============================================================

class LazyGenes:
    """
    Torch-like accessor over adata.X rows.

    Supports:
      genes[i]                -> (D,)
      genes[idx_array]        -> (B,D)
      genes[idx_array, :]     -> (B,D)
      genes[idx_array, cols]  -> (B,|cols|)
    """
    def __init__(self, X, global_rows: np.ndarray):
        self.X = X
        self.global_rows = np.asarray(global_rows, dtype=np.int64).reshape(-1)
        self.D = int(X.shape[1])

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return (int(self.global_rows.shape[0]), int(self.D))
        if dim == 0:
            return int(self.global_rows.shape[0])
        if dim == 1:
            return int(self.D)
        raise ValueError("dim must be 0 or 1")

    def __len__(self):
        return int(self.global_rows.shape[0])

    def _norm_rows(self, rsel):
        if isinstance(rsel, slice):
            return np.arange(len(self.global_rows), dtype=np.int64)[rsel], False
        if isinstance(rsel, (int, np.integer)):
            return np.array([int(rsel)], dtype=np.int64), True
        if torch.is_tensor(rsel):
            return rsel.detach().cpu().numpy().astype(np.int64).reshape(-1), False
        return np.asarray(rsel, dtype=np.int64).reshape(-1), False

    def _norm_cols(self, csel):
        if csel is None:
            return None, False
        if isinstance(csel, slice):
            if csel == slice(None, None, None):
                return None, False
            cols = np.arange(self.D, dtype=np.int64)[csel]
            return cols, False
        if isinstance(csel, (int, np.integer)):
            return np.array([int(csel)], dtype=np.int64), True
        if torch.is_tensor(csel):
            return csel.detach().cpu().numpy().astype(np.int64).reshape(-1), False
        return np.asarray(csel, dtype=np.int64).reshape(-1), False

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError("LazyGenes only supports genes[rows] or genes[rows, cols]")
            row_sel, col_sel = idx
        else:
            row_sel, col_sel = idx, None

        rows_local, row_scalar = self._norm_rows(row_sel)
        cols, col_scalar = self._norm_cols(col_sel)

        glob = self.global_rows[rows_local]

        if scipy.sparse.issparse(self.X):
            if cols is None:
                arr = self.X[glob].toarray().astype(np.float32)
            else:
                arr = self.X[glob][:, cols].toarray().astype(np.float32)
        else:
            if cols is None:
                arr = np.asarray(self.X[glob], dtype=np.float32)
            else:
                arr = np.asarray(self.X[np.ix_(glob, cols)], dtype=np.float32)

        out = torch.from_numpy(arr).float()

        if row_scalar and col_scalar:
            return out[0, 0]
        if row_scalar:
            return out[0]
        if col_scalar:
            return out[:, 0]
        return out


# ============================================================
#  Eval dataset object matching the evaluator API
# ============================================================

class GeneEvalDataset:
    """
    Minimal dataset API to satisfy gene_evaluate_*.
    """
    def __init__(
        self,
        adata,
        split_rows_global: np.ndarray,
        perturbations_all: torch.Tensor,
        covariates_all: List[torch.Tensor],
        cov_names_all: np.ndarray,
        pert_names_all: np.ndarray,
        cov_pert_all: np.ndarray,
        control_mask_all: np.ndarray,
        de_genes: Dict[str, List[str]],
        var_names,
    ):
        self.adata = adata
        self.global_rows = np.asarray(split_rows_global, dtype=np.int64).reshape(-1)

        self.genes = LazyGenes(adata.X, self.global_rows)

        self.perturbations = perturbations_all[self.global_rows].contiguous().cpu()
        self.covariates = [c[self.global_rows].contiguous().cpu() for c in covariates_all]

        self.cov_names = np.asarray(cov_names_all[self.global_rows], dtype=object)
        self.pert_names = np.asarray(pert_names_all[self.global_rows], dtype=object)
        self.cov_pert = np.asarray(cov_pert_all[self.global_rows], dtype=object)

        self.control = np.asarray(control_mask_all[self.global_rows], dtype=bool)

        self.de_genes = de_genes if isinstance(de_genes, dict) else {}
        self.var_names = var_names

    def __len__(self):
        return int(self.global_rows.shape[0])

    def subset_condition(self, control: Optional[bool] = None):
        if control is None:
            keep_local = np.arange(len(self), dtype=np.int64)
        else:
            keep_local = np.where(self.control == bool(control))[0].astype(np.int64)

        new_global_rows = self.global_rows[keep_local]
        return GeneEvalDataset(
            adata=self.adata,
            split_rows_global=new_global_rows,
            perturbations_all=self._perturbations_all,
            covariates_all=self._covariates_all,
            cov_names_all=self._cov_names_all,
            pert_names_all=self._pert_names_all,
            cov_pert_all=self._cov_pert_all,
            control_mask_all=self._control_mask_all,
            de_genes=self.de_genes,
            var_names=self.var_names,
        )

    def attach_globals(
        self,
        perturbations_all: torch.Tensor,
        covariates_all: List[torch.Tensor],
        cov_names_all: np.ndarray,
        pert_names_all: np.ndarray,
        cov_pert_all: np.ndarray,
        control_mask_all: np.ndarray,
    ):
        self._perturbations_all = perturbations_all
        self._covariates_all = covariates_all
        self._cov_names_all = cov_names_all
        self._pert_names_all = pert_names_all
        self._cov_pert_all = cov_pert_all
        self._control_mask_all = control_mask_all
        return self


# ============================================================
#  Evaluator utilities (standalone)
# ============================================================

def unique_ind(arr: np.ndarray) -> Dict[Any, np.ndarray]:
    out: Dict[Any, List[int]] = {}
    for i, v in enumerate(arr):
        out.setdefault(v, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}

def move_tensors(*args, device: str):
    moved = []
    for a in args:
        if isinstance(a, list):
            moved.append([t.to(device, non_blocking=True) for t in a])
        elif torch.is_tensor(a):
            moved.append(a.to(device, non_blocking=True))
        else:
            moved.append(a)
    return moved

def _r2(a, b) -> float:
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    if torch.is_tensor(b):
        b = b.detach().cpu().numpy()
    a = np.asarray(a)
    b = np.asarray(b)
    return float(r2_score(a, b))


# ============================================================
#  Evaluation code (Fix 2 applied in classic evaluator)
# ============================================================

def gene_evaluate(model, datasets, batch_size=None, eval_mode="native", **kwargs):
    if eval_mode == "native":
        return gene_evaluate_native(model, datasets, batch_size=batch_size)
    elif eval_mode == "classic":
        return gene_evaluate_classic(model, datasets, batch_size=batch_size)
    else:
        raise ValueError("eval_mode not recognized")

def gene_evaluate_native(model, datasets, batch_size=None):
    model.eval()
    with torch.no_grad():
        evaluation_stats = {
            "train": gene_evaluate_r2_native(
                model,
                datasets["train"].subset_condition(control=False),
                datasets["train"].subset_condition(control=None),
                batch_size=batch_size
            ),
            "test": gene_evaluate_r2_native(
                model,
                datasets["test"].subset_condition(control=False),
                datasets["train"].subset_condition(control=None),
                batch_size=batch_size
            ),
            "ood": gene_evaluate_r2_native(
                model,
                datasets["ood"],
                datasets["test"].subset_condition(control=None),
                batch_size=batch_size
            ),
        }
    model.train()
    return evaluation_stats, model.early_stopping(np.mean(evaluation_stats["test"]))

def gene_evaluate_r2_native(model, dataset, dataset_control, batch_size=None, min_samples=10):
    mean_score_mean, mean_score_robust = [], []
    mean_score_de_mean, mean_score_de_robust = [], []

    if len(dataset) == 0 or len(dataset_control) == 0:
        return [-1, -1, -1, -1]

    cov_cats = unique_ind(dataset.cov_names)
    cov_cats_control = unique_ind(dataset_control.cov_names)
    pert_cats = unique_ind(dataset.pert_names)

    for cov_category in cov_cats.keys():
        idx_control = cov_cats_control.get(cov_category, None)
        if idx_control is None or len(idx_control) == 0:
            continue

        genes_control = dataset_control.genes[idx_control]
        perts_control = dataset_control.perturbations[idx_control]
        covars_control = [covar[idx_control] for covar in dataset_control.covariates]

        pert_names_control = dataset_control.pert_names[idx_control]
        pert_names_control_cats = unique_ind(pert_names_control)

        num = genes_control.size(0)
        bs = num if batch_size is None else int(batch_size)

        for pert_category in pert_cats.keys():
            idx = np.intersect1d(cov_cats[cov_category], pert_cats[pert_category])
            if len(idx) > min_samples:
                cov_pert = dataset.cov_pert[idx[0]]
                de_genes = np.array(dataset.de_genes.get(cov_pert, []))
                de_idx = np.where(dataset.var_names.isin(de_genes))[0] if de_genes.size > 0 else np.array([])

                perts = dataset.perturbations[idx[0]].view(1, -1).repeat(num, 1).clone()

                num_eval = 0
                yp = []
                while num_eval < num:
                    end = min(num_eval + bs, num)
                    out = model.predict(*move_tensors(
                        genes_control[num_eval:end],
                        perts_control[num_eval:end],
                        [covar[num_eval:end] for covar in covars_control],
                        perts[num_eval:end],
                        device=model.device
                    ))
                    yp.append(out.detach().cpu())
                    num_eval += bs
                yp = torch.cat(yp, 0)

                yt = dataset.genes[idx, :]          # supported by LazyGenes (Fix 1)
                yt_m = yt.mean(0)

                yp_m = yp.mean(0)
                mean_score_mean.append(_r2(yt_m, yp_m))
                if de_idx.size > 0:
                    mean_score_de_mean.append(_r2(yt_m[de_idx], yp_m[de_idx]))

                if pert_category in pert_names_control_cats:
                    pert_idx = pert_names_control_cats[pert_category]
                    yp_r = yp_m + (genes_control[pert_idx] - yp[pert_idx]).mean(0)
                    mean_score_robust.append(_r2(yt_m, yp_r))
                    if de_idx.size > 0:
                        mean_score_de_robust.append(_r2(yt_m[de_idx], yp_r[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [
            mean_score_mean, mean_score_robust,
            mean_score_de_mean, mean_score_de_robust
        ]
    ]

def gene_evaluate_classic(model, datasets, batch_size=None):
    model.eval()
    with torch.no_grad():
        evaluation_stats = {
            "train": gene_evaluate_r2_classic(
                model,
                datasets["train"].subset_condition(control=False),
                datasets["train"].subset_condition(control=True),
                batch_size=batch_size
            ),
            "test": gene_evaluate_r2_classic(
                model,
                datasets["test"].subset_condition(control=False),
                datasets["test"].subset_condition(control=True),
                batch_size=batch_size
            ),
            "ood": gene_evaluate_r2_classic(
                model,
                datasets["ood"],
                datasets["test"].subset_condition(control=True),
                batch_size=batch_size
            ),
        }
    model.train()
    return evaluation_stats, model.early_stopping(np.mean(evaluation_stats["test"]))

def gene_evaluate_r2_classic(model, dataset, dataset_control, batch_size=None, min_samples=10):
    mean_score, mean_score_de = [], []

    if len(dataset) == 0 or len(dataset_control) == 0:
        return [-1, -1]

    # Fix 2: do NOT do dataset_control.genes[:] (would densify whole control split)
    genes_control = dataset_control.genes         # LazyGenes
    perts_control = dataset_control.perturbations
    num = genes_control.size(0)

    bs = num if batch_size is None else int(batch_size)

    for pert_category in np.unique(dataset.cov_pert):
        de_genes = np.array(dataset.de_genes.get(pert_category, []))
        de_idx = np.where(dataset.var_names.isin(de_genes))[0] if de_genes.size > 0 else np.array([])

        idx = np.where(dataset.cov_pert == pert_category)[0]
        if len(idx) > min_samples:
            perts = dataset.perturbations[idx][0].view(1, -1).repeat(num, 1).clone()
            covars = [
                covar[idx][0].view(1, -1).repeat(num, 1).clone()
                for covar in dataset.covariates
            ]

            num_eval = 0
            yp = []
            while num_eval < num:
                end = min(num_eval + bs, num)
                out = model.predict(*move_tensors(
                    genes_control[num_eval:end],                # lazy slice
                    perts_control[num_eval:end],
                    [covar[num_eval:end] for covar in covars],
                    perts[num_eval:end],
                    device=model.device
                ))
                yp.append(out.detach().cpu())
                num_eval += bs
            yp = torch.cat(yp, 0)
            yp_m = yp.mean(0)

            yt = dataset.genes[idx, :].detach().cpu().numpy()
            yt_m = yt.mean(axis=0)

            mean_score.append(_r2(yt_m, yp_m))
            if de_idx.size > 0:
                mean_score_de.append(_r2(yt_m[de_idx], yp_m[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]


# ============================================================
#  Your UV model adapted to gene setting
# ============================================================

def kl_diag_gauss_per_dim(q_mu, q_lv, p_mu, p_lv) -> torch.Tensor:
    q_var = torch.exp(q_lv)
    p_var = torch.exp(p_lv)
    return 0.5 * (p_lv - q_lv + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) - 1.0)

def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def gaussian_nll_diag(y, mu, logvar) -> torch.Tensor:
    return (0.5 * (math.log(2.0 * math.pi) + logvar + (y - mu) ** 2 / (torch.exp(logvar) + 1e-8))).mean()

def free_bits_stopgrad(kl_bxd: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        return kl_bxd.sum(dim=-1).mean()
    kl_dim = kl_bxd.mean(dim=0)
    kl_dim_eff = torch.where(kl_dim >= tau, kl_dim, kl_dim.detach() * 0.0 + float(tau))
    return kl_dim_eff.sum()

def rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((n, n), device=x.device)
    x2 = (x * x).sum(dim=1, keepdim=True)
    d2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    sigma2 = max(float(sigma) ** 2, 1e-8)
    return torch.exp(-d2 / (2.0 * sigma2))

def median_heuristic_sigma(x: torch.Tensor, max_points: int = 256) -> float:
    n = x.shape[0]
    if n <= 2:
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

def hsic_rbf(x: torch.Tensor, z: torch.Tensor, sigma_x: float = -1.0, sigma_z: float = -1.0) -> torch.Tensor:
    n = x.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=x.device)
    if sigma_x < 0:
        sigma_x = median_heuristic_sigma(x.detach())
    if sigma_z < 0:
        sigma_z = median_heuristic_sigma(z.detach())
    K = rbf_kernel(x, sigma_x)
    L = rbf_kernel(z, sigma_z)
    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH).sum() / ((n - 1) * (n - 1))
    return hsic.clamp_min(0.0)

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
    def __init__(self, x_dim: int, u_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        self.backbone = MLP(x_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, u_dim)
        self.lv = nn.Linear(hidden, u_dim)

    def forward(self, x):
        h = self.backbone(x)
        lv = self.lv(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class PriorV(nn.Module):
    def __init__(self, x_dim: int, a_dim: int, a_emb: int, v_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, a_emb)
        self.backbone = MLP(x_dim + a_emb, hidden, hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, v_dim)
        self.lv = nn.Linear(hidden, v_dim)

    def forward(self, x, a):
        ae = F.relu(self.a_proj(a))
        h = self.backbone(torch.cat([x, ae], dim=-1))
        lv = self.lv(h).clamp(-6.0, 6.0)
        return self.mu(h), lv

class EncoderUV(nn.Module):
    def __init__(self, x_dim: int, a_dim: int, y_dim: int, a_emb: int, y_emb: int,
                 u_dim: int, v_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, a_emb)
        self.y_enc = nn.Sequential(
            nn.Linear(y_dim, y_emb),
            nn.ReLU(inplace=True),
            nn.Linear(y_emb, y_emb),
            nn.ReLU(inplace=True),
        )
        self.backbone = MLP(x_dim + a_emb + y_emb, hidden, hidden, depth=depth, dropout=dropout)
        self.u_mu = nn.Linear(hidden, u_dim)
        self.u_lv = nn.Linear(hidden, u_dim)
        self.v_mu = nn.Linear(hidden, v_dim)
        self.v_lv = nn.Linear(hidden, v_dim)

    def forward(self, x, a, y):
        ae = F.relu(self.a_proj(a))
        ye = self.y_enc(y)
        h = self.backbone(torch.cat([x, ae, ye], dim=-1))
        u_lv = self.u_lv(h).clamp(-6.0, 6.0)
        v_lv = self.v_lv(h).clamp(-6.0, 6.0)
        return self.u_mu(h), u_lv, self.v_mu(h), v_lv

class Decoder(nn.Module):
    def __init__(self, x_dim: int, a_dim: int, a_emb: int, y_dim: int,
                 u_dim: int, v_dim: int, hidden: int, depth: int, dropout: float,
                 y_lv_min: float = -6.0, y_lv_max: float = 3.0):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, a_emb)
        self.trunk = MLP(x_dim + u_dim, hidden, hidden, depth=depth, dropout=dropout)
        self.head = MLP(hidden + v_dim + a_emb, hidden, hidden, depth=2, dropout=dropout)
        self.mu = nn.Linear(hidden, y_dim)
        self.lv = nn.Linear(hidden, y_dim)
        self.y_lv_min = float(y_lv_min)
        self.y_lv_max = float(y_lv_max)

    def forward(self, x, a, u, v):
        ae = F.relu(self.a_proj(a))
        h0 = self.trunk(torch.cat([x, u], dim=-1))
        h = self.head(torch.cat([h0, v, ae], dim=-1))
        mu = self.mu(h)
        lv = self.lv(h).clamp(self.y_lv_min, self.y_lv_max)
        return mu, lv

class UVTwinGeneModel(nn.Module):
    """
    Wrapper to match evaluator API: .predict(genes_control, perts_control, covars_control, perts_target)
    """
    def __init__(self, x_dim, a_dim, y_dim, cfg):
        super().__init__()
        self.device = cfg.device
        self.prior_u = PriorU(x_dim, cfg.u_dim, cfg.hidden, cfg.depth, cfg.dropout)
        self.prior_v = PriorV(x_dim, a_dim, cfg.a_emb, cfg.v_dim, cfg.hidden, cfg.depth, cfg.dropout)
        self.enc = EncoderUV(x_dim, a_dim, y_dim, cfg.a_emb, cfg.y_emb, cfg.u_dim, cfg.v_dim, cfg.hidden, cfg.depth, cfg.dropout)
        self.dec = Decoder(x_dim, a_dim, cfg.a_emb, y_dim, cfg.u_dim, cfg.v_dim, cfg.hidden, cfg.depth, cfg.dropout)
        self.to(self.device)

    def early_stopping(self, score: float) -> bool:
        return False

    @torch.no_grad()
    def predict(self, genes_control, perts_control, covars_control_list, perts_target):
        x = torch.cat(covars_control_list, dim=1)
        y0 = genes_control
        a0 = perts_control
        at = perts_target

        mu_u_q, _lv_u_q, _mu_v_q, _lv_v_q = self.enc(x, a0, y0)
        u = mu_u_q

        mu_v_t, _lv_v_t = self.prior_v(x, at)
        v = mu_v_t

        mu, _lv = self.dec(x, at, u, v)
        return mu


# ============================================================
#  Training dataset (for SGD loop)
# ============================================================

class GeneTrainDataset(Dataset):
    """
    Returns:
      y:  (D,) float32
      a:  (A,) float32
      covars: list of (Ck,) float32 each
      a_cf: (A,) float32 sampled from pool
    """
    def __init__(
        self,
        X,
        rows_global: np.ndarray,
        perturbations_all: torch.Tensor,
        covariates_all: List[torch.Tensor],
        allow_control_cf: bool,
        control_mask_all: np.ndarray,
        pert_names_all: np.ndarray,
        seed: int = 0,
    ):
        self.X = X
        self.rows = np.asarray(rows_global, dtype=np.int64).reshape(-1)
        self.A = int(perturbations_all.shape[1])
        self.D = int(X.shape[1])

        self.pert_all = perturbations_all
        self.cov_all = covariates_all

        self.allow_control_cf = bool(allow_control_cf)
        self.control_mask_all = np.asarray(control_mask_all, dtype=bool)
        self.pert_names_all = np.asarray(pert_names_all, dtype=object)

        rng = np.random.RandomState(seed)
        if self.allow_control_cf:
            pool = np.arange(len(self.rows), dtype=np.int64)
        else:
            glob = self.rows
            pool = np.where(~self.control_mask_all[glob])[0].astype(np.int64)
            if pool.size == 0:
                pool = np.arange(len(self.rows), dtype=np.int64)
        self._pool_local = pool
        self._rng = rng

    def __len__(self):
        return int(self.rows.shape[0])

    def __getitem__(self, j: int):
        i = int(self.rows[j])
        y = safe_to_dense_row(self.X, i)
        y_t = torch.from_numpy(y).float()

        a = self.pert_all[i].float()
        covs = [c[i].float() for c in self.cov_all]

        jj = int(self._pool_local[self._rng.randint(0, len(self._pool_local))])
        i_cf = int(self.rows[jj])
        a_cf = self.pert_all[i_cf].float()

        return y_t, a, covs, a_cf

def collate_train(batch):
    ys, a, covs, a_cf = zip(*batch)
    y = torch.stack(ys, 0)
    a = torch.stack(a, 0)
    a_cf = torch.stack(a_cf, 0)
    n_cov = len(covs[0])
    cov_list = []
    for k in range(n_cov):
        cov_list.append(torch.stack([covs[i][k] for i in range(len(covs))], 0))
    return y, a, cov_list, a_cf


# ============================================================
#  Training loop (RUN 1: AdamW + weight_decay + LR schedule + dropout=0.2)
# ============================================================

@dataclass
class TrainCfg:
    device: str = "cpu"
    epochs: int = 200
    batch_size: int = 128

    # RUN 1
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_milestones: Tuple[int, int] = (50, 120)
    lr_gamma: float = 0.5

    u_dim: int = 64
    v_dim: int = 64
    a_emb: int = 64
    y_emb: int = 256
    hidden: int = 512
    depth: int = 2
    dropout: float = 0.2   

    beta_kl: float = 1.0
    kl_anneal_epochs: int = 100
    free_bits_tau: float = 0.003

    lambda_lat: float = 0.1
    lambda_balance: float = 0.5

    log_every: int = 10
    eval_every: int = 10
    eval_mode: str = "both"
    eval_batch_size: int = 256

def train_loop(
    model: UVTwinGeneModel,
    train_loader: DataLoader,
    datasets_eval: Dict[str, GeneEvalDataset],
    cfg: TrainCfg,
):
    # RUN 1: AdamW + weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # RUN 1: LR schedule: 2e-4 -> 1e-4 at epoch 51, -> 5e-5 at epoch 121
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=list(cfg.lr_milestones),
        gamma=cfg.lr_gamma
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta = cfg.beta_kl * min(1.0, epoch / max(1, cfg.kl_anneal_epochs))

        loss_sum = rec_sum = klu_sum = klv_sum = lat_sum = bal_sum = 0.0
        nb = 0

        for y, a, covs, a_cf in train_loader:
            y = y.to(cfg.device, non_blocking=True)
            a = a.to(cfg.device, non_blocking=True)
            a_cf = a_cf.to(cfg.device, non_blocking=True)
            covs = [c.to(cfg.device, non_blocking=True) for c in covs]
            x = torch.cat(covs, dim=1)

            mu_u_q, lv_u_q, mu_v_q, lv_v_q = model.enc(x, a, y)
            u = reparam(mu_u_q, lv_u_q)
            v = reparam(mu_v_q, lv_v_q)

            mu_u_p, lv_u_p = model.prior_u(x)
            mu_v_p, lv_v_p = model.prior_v(x, a)

            mu_hat, lv_hat = model.dec(x, a, u, v)
            recon = gaussian_nll_diag(y, mu_hat, lv_hat)

            kl_u_bxd = kl_diag_gauss_per_dim(mu_u_q, lv_u_q, mu_u_p, lv_u_p)
            kl_v_bxd = kl_diag_gauss_per_dim(mu_v_q, lv_v_q, mu_v_p, lv_v_p)
            kl_u = free_bits_stopgrad(kl_u_bxd, cfg.free_bits_tau)
            kl_v = free_bits_stopgrad(kl_v_bxd, cfg.free_bits_tau)

            lat = torch.tensor(0.0, device=cfg.device)
            if cfg.lambda_lat > 0:
                mv_cf, lv_cf = model.prior_v(x, a_cf)
                v_cf = reparam(mv_cf, lv_cf)
                mu_ycf, _lv_ycf = model.dec(x, a_cf, u, v_cf)
                mu_u_cf, _lv_u_cf, _mv2, _lv2 = model.enc(x, a_cf, mu_ycf.detach())
                lat = F.mse_loss(mu_u_cf, mu_u_q.detach())

            bal = torch.tensor(0.0, device=cfg.device)
            if cfg.lambda_balance > 0:
                bal = hsic_rbf(mu_u_q, a)

            loss = recon + beta * (kl_u + kl_v) + cfg.lambda_lat * lat + cfg.lambda_balance * bal

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_sum += float(loss.item())
            rec_sum += float(recon.item())
            klu_sum += float(kl_u.item())
            klv_sum += float(kl_v.item())
            lat_sum += float(lat.item())
            bal_sum += float(bal.item())
            nb += 1

        # Step LR scheduler at END of epoch:
        # - milestone=50 triggers after epoch 50 -> epoch 51 uses decayed LR (2e-4 * 0.5 = 1e-4)
        # - milestone=120 triggers after epoch 120 -> epoch 121 uses 5e-5
        scheduler.step()

        if epoch % cfg.log_every == 0 or epoch == 1:
            cur_lr = float(opt.param_groups[0]["lr"])
            print(
                f"Epoch {epoch:04d} | "
                f"Loss={loss_sum/max(1,nb):.4f} "
                f"ReconNLL={rec_sum/max(1,nb):.4f} "
                f"KLu={klu_sum/max(1,nb):.4f} "
                f"KLv={klv_sum/max(1,nb):.4f} "
                f"Lat(u)={lat_sum/max(1,nb):.6f} "
                f"HSIC={bal_sum/max(1,nb):.6f} "
                f"| beta={beta:.3f} lr={cur_lr:.2e} wd={cfg.weight_decay:.1e} drop={cfg.dropout:.2f}"
            )

        if epoch % cfg.eval_every == 0 or epoch == 1:
            if cfg.eval_mode in ("native", "both"):
                stats, _ = gene_evaluate_native(model, datasets_eval, batch_size=cfg.eval_batch_size)
                print("[Eval-native] train:", stats["train"], "test:", stats["test"], "ood:", stats["ood"])
            if cfg.eval_mode in ("classic", "both"):
                stats, _ = gene_evaluate_classic(model, datasets_eval, batch_size=cfg.eval_batch_size)
                print("[Eval-classic] train:", stats["train"], "test:", stats["test"], "ood:", stats["ood"])


# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad_path", type=str, default="data/sciplex/sciplex_prepped.h5ad")
    p.add_argument("--mode", type=str, default="train", choices=["test_loader", "train"])

    p.add_argument("--perturbation_key", type=str, default="perturbation")
    p.add_argument("--control_key", type=str, default="control")
    p.add_argument("--dose_key", type=str, default="dose")
    p.add_argument("--covariate_keys", type=str, default="covariates",
                   help="Either 'covariates' to read adata.uns['fields']['covariates'], or comma-separated obs columns.")
    p.add_argument("--split_key", type=str, default="split")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--allow_control_cf", action="store_true")

    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=2e-4)             # RUN 1
    p.add_argument("--weight_decay", type=float, default=1e-4)    # RUN 1
    p.add_argument("--dropout", type=float, default=0.2)          # RUN 1
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval_mode", type=str, default="both", choices=["native", "classic", "both"])
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=512)

    args = p.parse_args()
    seed_everything(args.seed)

    adata = sc.read(args.h5ad_path)
    inspect_adata_schema(adata)

    perturbation_key = resolve_field(adata, args.perturbation_key, default="perturbation")
    control_key = resolve_field(adata, args.control_key, default="control")
    dose_key = resolve_field(adata, args.dose_key, default="dose")
    split_key = resolve_field(adata, args.split_key, default="split")

    if args.covariate_keys.strip() == "covariates":
        cov = resolve_field(adata, "covariates", default="covariates")
        covariate_keys = as_list_of_str(cov)
    else:
        covariate_keys = [c.strip() for c in args.covariate_keys.split(",") if c.strip()]

    if isinstance(covariate_keys, np.ndarray):
        covariate_keys = covariate_keys.tolist()
    covariate_keys = [str(k) for k in covariate_keys]

    if perturbation_key not in adata.obs.columns:
        raise KeyError(f"perturbation_key '{perturbation_key}' not found in adata.obs.")
    if control_key not in adata.obs.columns:
        raise KeyError(f"control_key '{control_key}' not found in adata.obs.")
    if split_key not in adata.obs.columns:
        raise KeyError(f"split_key '{split_key}' not found in adata.obs.")
    if dose_key is not None and dose_key not in adata.obs.columns:
        print(f"[Warn] dose_key '{dose_key}' not found in obs; using dummy 1.0 dose.")
        dose_key = None
    for k in covariate_keys:
        assert k in adata.obs.columns, f"Covariate '{k}' missing in adata.obs"

    print("[Keys] perturbation_key =", perturbation_key)
    print("[Keys] control_key      =", control_key)
    print("[Keys] dose_key         =", dose_key)
    print("[Keys] covariate_keys   =", covariate_keys)
    print("[Keys] split_key        =", split_key)

    N = int(adata.n_obs)
    pert_names_raw = adata.obs[perturbation_key].astype(str).values
    if dose_key is None:
        dose_vals = np.ones((N,), dtype=np.float32)
    else:
        dose_vals = adata.obs[dose_key].values

    atoms = []
    for pstr in pert_names_raw:
        atoms.extend(parse_combo_string(pstr))
    atoms = list(dict.fromkeys(atoms))
    atom_to_idx = {a: i for i, a in enumerate(atoms)}
    A = len(atoms)

    a_mat = np.zeros((N, A), dtype=np.float32)
    for i in range(N):
        p_list = parse_combo_string(pert_names_raw[i])
        d_list = parse_dose_combo(dose_vals[i])
        if len(d_list) == 1 and len(p_list) > 1:
            d_list = d_list * len(p_list)
        if len(d_list) != len(p_list):
            if len(d_list) < len(p_list):
                d_list = d_list + [d_list[-1]] * (len(p_list) - len(d_list))
            else:
                d_list = d_list[:len(p_list)]
        for p_name, d in zip(p_list, d_list):
            if p_name in atom_to_idx:
                a_mat[i, atom_to_idx[p_name]] += float(d)

    perturbations_all = torch.from_numpy(a_mat).float()

    covariates_all: List[torch.Tensor] = []
    cov_cols_as_str = []
    for ck in covariate_keys:
        vals = adata.obs[ck].astype(str).values
        cov_cols_as_str.append(vals)
        ohe, _mapping = one_hot_from_categories(vals)
        covariates_all.append(ohe)

    cov_names_all = np.array(["_".join(z) for z in zip(*cov_cols_as_str)], dtype=object)

    if dose_key is None:
        dose_str = np.array(["1.0"] * N, dtype=object)
    else:
        dose_str = np.array([str(d) for d in dose_vals], dtype=object)

    pert_dose = np.array([f"{pert_names_raw[i]}_{dose_str[i]}" for i in range(N)], dtype=object)
    pert_names_all = pert_dose
    cov_pert_all = np.array([f"{cov_names_all[i]}_{pert_names_all[i]}" for i in range(N)], dtype=object)

    control_vals = adata.obs[control_key].values
    control_mask_all = np.array([_to_int01(v) == 1 for v in control_vals], dtype=bool)

    de_genes = {}
    if "de_genes" in adata.uns and isinstance(adata.uns["de_genes"], dict):
        de_genes = adata.uns["de_genes"]
        print(f"[DE] Loaded de_genes from adata.uns['de_genes'] with {len(de_genes)} keys.")
    else:
        print("[DE] No adata.uns['de_genes'] found; DE-based metrics will be -1.")

    split_col = adata.obs[split_key].astype(str).values
    rows_train = np.where(split_col == "train")[0].astype(np.int64)
    rows_test  = np.where(split_col == "test")[0].astype(np.int64)
    rows_ood   = np.where(split_col == "ood")[0].astype(np.int64)

    if rows_train.size == 0:
        raise ValueError("No 'train' split found in obs[split].")

    ds_train_eval = GeneEvalDataset(
        adata, rows_train, perturbations_all, covariates_all,
        cov_names_all, pert_names_all, cov_pert_all,
        control_mask_all, de_genes, adata.var_names
    ).attach_globals(perturbations_all, covariates_all, cov_names_all, pert_names_all, cov_pert_all, control_mask_all)

    ds_test_eval = GeneEvalDataset(
        adata, rows_test, perturbations_all, covariates_all,
        cov_names_all, pert_names_all, cov_pert_all,
        control_mask_all, de_genes, adata.var_names
    ).attach_globals(perturbations_all, covariates_all, cov_names_all, pert_names_all, cov_pert_all, control_mask_all)

    ds_ood_eval = GeneEvalDataset(
        adata, rows_ood, perturbations_all, covariates_all,
        cov_names_all, pert_names_all, cov_pert_all,
        control_mask_all, de_genes, adata.var_names
    ).attach_globals(perturbations_all, covariates_all, cov_names_all, pert_names_all, cov_pert_all, control_mask_all)

    datasets_eval = {"train": ds_train_eval, "test": ds_test_eval, "ood": ds_ood_eval}

    print(f"[Splits] train={len(ds_train_eval)} test={len(ds_test_eval)} ood={len(ds_ood_eval)}")
    print(f"[Dims] y_dim(D)={adata.n_vars} a_dim(A)={A} x_dim={sum(int(c.shape[1]) for c in covariates_all)}")

    if args.mode == "test_loader":
        ds_train = GeneTrainDataset(
            X=adata.X,
            rows_global=rows_train,
            perturbations_all=perturbations_all,
            covariates_all=covariates_all,
            allow_control_cf=bool(args.allow_control_cf),
            control_mask_all=control_mask_all,
            pert_names_all=pert_names_all,
            seed=args.seed,
        )
        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_train)
        y, a, covs, a_cf = next(iter(loader))
        x = torch.cat(covs, dim=1)
        print("[Batch] y:", tuple(y.shape), "a:", tuple(a.shape), "x:", tuple(x.shape), "a_cf:", tuple(a_cf.shape))
        print("[Batch] a nonzero per row (avg):", float((a.abs() > 1e-8).float().sum(dim=1).mean()))
        return

    x_dim = sum(int(c.shape[1]) for c in covariates_all)
    y_dim = int(adata.n_vars)

    cfg = TrainCfg(
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        eval_every=int(args.eval_every),
        eval_mode=str(args.eval_mode),
        eval_batch_size=int(args.eval_batch_size),
    )

    model = UVTwinGeneModel(x_dim=x_dim, a_dim=A, y_dim=y_dim, cfg=cfg)

    ds_train = GeneTrainDataset(
        X=adata.X,
        rows_global=rows_train,
        perturbations_all=perturbations_all,
        covariates_all=covariates_all,
        allow_control_cf=bool(args.allow_control_cf),
        control_mask_all=control_mask_all,
        pert_names_all=pert_names_all,
        seed=args.seed,
    )
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
        drop_last=False,
    )

    train_loop(model, train_loader, datasets_eval, cfg)


if __name__ == "__main__":
    main()








# data/sciplex/sciplex_prepped.h5ad