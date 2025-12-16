import math
import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskMemory:
    U: torch.Tensor      # [out_features, k]
    sigma: torch.Tensor  # [k]
    V: torch.Tensor      # [in_features, k]
    task_id: str
    
    @property
    def rank(self) -> int:
        return len(self.sigma)
    
    def to(self, device: torch.device) -> 'TaskMemory':
        return TaskMemory(
            U=self.U.to(device),
            sigma=self.sigma.to(device),
            V=self.V.to(device),
            task_id=self.task_id
        )

@dataclass
class SynthesizedSubspaces:
    V_reuse: Optional[torch.Tensor] = None    # Q1+Q2 输入空间
    V_protect: Optional[torch.Tensor] = None  # Q3+Q4 输入空间
    U_reuse: Optional[torch.Tensor] = None    # Q1+Q3 输出空间
    U_protect: Optional[torch.Tensor] = None  # Q2+Q4 输出空间
    
    friends: List[str] = field(default_factory=list)    # Q1
    enemies: List[str] = field(default_factory=list)    # Q2
    benign: List[str] = field(default_factory=list)     # Q3
    strangers: List[str] = field(default_factory=list)  # Q4
    
    @property
    def similar_tasks(self) -> List[str]:
        return self.friends + self.enemies + self.benign
    
    @property
    def different_tasks(self) -> List[str]:
        return self.strangers


def truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_dtype = matrix.dtype
    mat_f32 = matrix.detach().float()
    
    U_full, S_full, Vh_full = torch.linalg.svd(mat_f32, full_matrices=False)
    
    k = min(rank, len(S_full), U_full.shape[1])
    U = U_full[:, :k]
    sigma = S_full[:k]
    V = Vh_full[:k, :].T
    
    mask = sigma > eps
    if mask.sum() < k and mask.sum() > 0:
        valid_k = int(mask.sum().item())
        U, sigma, V = U[:, :valid_k], sigma[:valid_k], V[:, :valid_k]
    
    return U.to(orig_dtype), sigma.to(orig_dtype), V.to(orig_dtype)


def stable_qr_orthogonalize(
    matrices: List[torch.Tensor],
    weights: Optional[List[torch.Tensor]] = None,
    rank_threshold: float = 1e-5
) -> Optional[torch.Tensor]:
    if not matrices:
        return None
    
    valid_pairs = []
    for i, m in enumerate(matrices):
        if m is not None and m.numel() > 0:
            w = weights[i] if (weights is not None and i < len(weights)) else None
            valid_pairs.append((m, w))
            
    if not valid_pairs:
        return None
        
    valid_m = [p[0] for p in valid_pairs]
    valid_w = [p[1] for p in valid_pairs]
    
    orig_dtype = valid_m[0].dtype
    device = valid_m[0].device
    
    valid_m_f32 = [m.float().to(device) for m in valid_m]
    
    use_weights = any(w is not None and w.numel() > 0 for w in valid_w)
    
    if use_weights:
        weighted_list = []
        for m, w in zip(valid_m_f32, valid_w):
            w_vec = torch.ones(m.shape[1], device=device, dtype=torch.float32)
            
            if w is not None and w.numel() > 0:
                w_f32 = w.float().to(device)
                len_w = min(len(w_f32), m.shape[1])
                w_vec[:len_w] = w_f32[:len_w]
                w_vec = torch.clamp(w_vec, min=1e-6)
                w_vec = w_vec / w_vec.max()
            
            weighted_list.append(m * w_vec.unsqueeze(0))
            
        concat = torch.cat(weighted_list, dim=1)
    else:
        concat = torch.cat(valid_m_f32, dim=1)

    # QR
    Q, R = torch.linalg.qr(concat)
    
    diag_R = torch.abs(torch.diag(R))
    if diag_R.max() > 0:
        mask = diag_R > rank_threshold * diag_R.max()
        effective_rank = max(1, mask.sum().item())
        Q = Q[:, :effective_rank]
    
    return Q.to(orig_dtype)


def bilateral_similarity(
    W_probe: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[float, float]:
    W = W_probe.detach().float()
    U_f, V_f = U.float(), V.float()
    
    W_norm_sq = torch.norm(W, p='fro') ** 2 + eps
    rho_in = (torch.norm(W @ V_f, p='fro') ** 2 / W_norm_sq).item()
    rho_out = (torch.norm(U_f.T @ W, p='fro') ** 2 / W_norm_sq).item()
    
    return rho_in, rho_out


def route_gradient_B(
    g_B: torch.Tensor,
    U_protect: Optional[torch.Tensor],
    U_reuse: Optional[torch.Tensor],
    boost_factor: float = 0.5,
    damp_factor: float = 0.8,
) -> torch.Tensor:
    g = g_B.clone()
    
    if not torch.isfinite(g).all():
        return torch.zeros_like(g_B)
    
    if U_protect is None and U_reuse is None:
        return g
    
    # 投影到 protect
    if U_protect is not None and U_protect.shape[1] > 0:
        U_p = U_protect.to(dtype=g.dtype, device=g.device)
        g_protect = U_p @ (U_p.T @ g)
    else:
        g_protect = torch.zeros_like(g)
    
    g_rest = g - g_protect
    if U_reuse is not None and U_reuse.shape[1] > 0:
        U_r = U_reuse.to(dtype=g.dtype, device=g.device)
        g_reuse = U_r @ (U_r.T @ g_rest)
    else:
        g_reuse = torch.zeros_like(g)
    
    g_orth = g_rest - g_reuse
    
    g_final = (1.0 - damp_factor) * g_protect + (1.0 + boost_factor) * g_reuse + g_orth
    
    return g_final if torch.isfinite(g_final).all() else g


def route_gradient_A(
    g_A: torch.Tensor,
    V_protect: Optional[torch.Tensor],
    V_reuse: Optional[torch.Tensor],
    boost_factor: float = 0.5,
    damp_factor: float = 0.8
) -> torch.Tensor:
    g = g_A.clone()
    
    if not torch.isfinite(g).all():
        return torch.zeros_like(g_A)
    
    if V_protect is None and V_reuse is None:
        return g
    
    if V_protect is not None and V_protect.shape[1] > 0:
        V_p = V_protect.to(dtype=g.dtype, device=g.device)
        g_protect = g @ V_p @ V_p.T
    else:
        g_protect = torch.zeros_like(g)
    
    g_rest = g - g_protect
    if V_reuse is not None and V_reuse.shape[1] > 0:
        V_r = V_reuse.to(dtype=g.dtype, device=g.device)
        g_reuse = g_rest @ V_r @ V_r.T
    else:
        g_reuse = torch.zeros_like(g)
    
    g_orth = g_rest - g_reuse
    
    g_final = (1.0 - damp_factor) * g_protect + (1.0 + boost_factor) * g_reuse + g_orth
    
    return g_final if torch.isfinite(g_final).all() else g


def pad_orthogonal_columns(matrix: torch.Tensor, target_cols: int) -> torch.Tensor:
    d, k = matrix.shape
    if k >= target_cols:
        return matrix[:, :target_cols]
    
    candidates = torch.randn(d, target_cols - k, device=matrix.device, dtype=torch.float32)
    combined = torch.cat([matrix.float(), candidates], dim=1)
    Q, _ = torch.linalg.qr(combined)
    
    return Q[:, :target_cols].to(matrix.dtype)