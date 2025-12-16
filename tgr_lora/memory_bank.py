import os
import json
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from .math_utils import TaskMemory, SynthesizedSubspaces, stable_qr_orthogonalize, bilateral_similarity
from .config import TGRLoRAConfig

logger = logging.getLogger(__name__)


class MemoryBank:
    def __init__(self, config: TGRLoRAConfig):
        self.config = config
        self._storage: Dict[str, Dict[str, TaskMemory]] = defaultdict(dict)
        self._task_order: List[str] = []
        self._device = torch.device('cpu')
    
    def set_device(self, device: torch.device):
        self._device = device
    
    @property
    def num_tasks(self) -> int:
        return len(self._task_order)
    
    def get_layer_memories(self, layer: str) -> Dict[str, TaskMemory]:
        return dict(self._storage.get(layer, {}))

    def add_memory(
        self,
        layer: str,
        task_id: str,
        B: torch.Tensor,
        A: torch.Tensor,
        rank: Optional[int] = None
    ) -> TaskMemory:
        if rank is None:
            rank = self.config.rank
        
        with torch.no_grad():
            B_f32, A_f32 = B.float(), A.float()
            
            # QR + SVD
            Q_b, R_b = torch.linalg.qr(B_f32)
            Q_a, R_a = torch.linalg.qr(A_f32.T)
            
            M = R_b @ R_a.T
            U_small, sigma, Vt_small = torch.linalg.svd(M)
            
            U = Q_b @ U_small
            V = Q_a @ Vt_small.T
            
            k = min(rank, len(sigma))
            U, sigma, V = U[:, :k], sigma[:k], V[:, :k]
        
        memory = TaskMemory(
            U=U.to(self._device),
            sigma=sigma.to(self._device),
            V=V.to(self._device),
            task_id=task_id
        )
        
        self._storage[layer][task_id] = memory
        if task_id not in self._task_order:
            self._task_order.append(task_id)
        
        # logger.info(f"Added memory: {task_id} @ {layer}, rank={memory.rank}")
        return memory
    
    def compute_similarities(
        self,
        layer: str,
        W_probe: torch.Tensor
    ) -> Dict[str, Tuple[float, float]]:
        W_probe = W_probe.to(self._device)
        sims = {}
        for task_id, mem in self.get_layer_memories(layer).items():
            rho_in, rho_out = bilateral_similarity(
                W_probe, mem.U.to(self._device), mem.V.to(self._device)
            )
            sims[task_id] = (rho_in, rho_out)
        return sims
    
    def partition_tasks(
        self,
        sims: Dict[str, Tuple[float, float]],
        tau_in: float,
        tau_out: float
    ) -> Dict[str, List[str]]:
        groups = {'friends': [], 'enemies': [], 'benign': [], 'strangers': []}
        
        for task_id, (rho_in, rho_out) in sims.items():
            high_in, high_out = rho_in > tau_in, rho_out > tau_out
            
            if high_in and high_out:
                groups['friends'].append(task_id)
            elif high_in:
                groups['enemies'].append(task_id)
            elif high_out:
                groups['benign'].append(task_id)
            else:
                groups['strangers'].append(task_id)

        return groups
    
    def synthesize_subspaces(
        self,
        layer: str,
        groups: Dict[str, List[str]],
        use_weights: bool = True
    ) -> SynthesizedSubspaces:
        mems = self.get_layer_memories(layer)
        
        def collect(task_ids):
            Us, Vs, sigmas = [], [], []
            for tid in task_ids:
                if tid in mems:
                    m = mems[tid]
                    Us.append(m.U.to(self._device))
                    Vs.append(m.V.to(self._device))
                    sigmas.append(m.sigma.to(self._device) if use_weights else None)
            return Us, Vs, sigmas
        
        # V_reuse: Q1+Q2, V_protect: Q3+Q4
        _, Vs_reuse, sigs = collect(groups['friends'] + groups['enemies'])
        V_reuse = stable_qr_orthogonalize(Vs_reuse, sigs if use_weights else None)
        
        _, Vs_protect, sigs = collect(groups['benign'] + groups['strangers'])
        V_protect = stable_qr_orthogonalize(Vs_protect, sigs if use_weights else None)
        
        # U_reuse: Q1+Q3, U_protect: Q2+Q4
        Us_reuse, _, sigs = collect(groups['friends'] + groups['benign'])
        U_reuse = stable_qr_orthogonalize(Us_reuse, sigs if use_weights else None)
        
        Us_protect, _, sigs = collect(groups['enemies'] + groups['strangers'])
        U_protect = stable_qr_orthogonalize(Us_protect, sigs if use_weights else None)
        
        return SynthesizedSubspaces(
            V_reuse=V_reuse, V_protect=V_protect,
            U_reuse=U_reuse, U_protect=U_protect,
            friends=groups['friends'], enemies=groups['enemies'],
            benign=groups['benign'], strangers=groups['strangers']
        )
    
    def synthesize_all_layers(
        self,
        probe_updates: Dict[str, torch.Tensor],
        tau_in: Optional[float] = None,
        tau_out: Optional[float] = None
    ) -> Dict[str, SynthesizedSubspaces]:
        tau_in = tau_in or self.config.similarity_threshold_in
        tau_out = tau_out or self.config.similarity_threshold_out
        logger.info(f'RHO IN: {tau_in}, RHO OUT: {tau_out}')
        result = {}
        for layer, W_probe in probe_updates.items():
            logger.info(f'Layer: {layer}')
            # {task1: (rho in, rho out), ...}
            sims = self.compute_similarities(layer, W_probe)
            logger.info(f'sims: {sims}')
            groups = self.partition_tasks(sims, tau_in, tau_out)
            logger.info(f'groups: {groups}')
            result[layer] = self.synthesize_subspaces(layer, groups)
        return result

        
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump({'task_order': self._task_order, 'layers': list(self._storage.keys())}, f)
        
        for layer, mems in self._storage.items():
            layer_dir = os.path.join(path, layer.replace('.', '_'))
            os.makedirs(layer_dir, exist_ok=True)
            for task_id, mem in mems.items():
                torch.save({
                    'U': mem.U.cpu(), 'sigma': mem.sigma.cpu(),
                    'V': mem.V.cpu(), 'task_id': mem.task_id
                }, os.path.join(layer_dir, f'{task_id}.pt'))
    
    def load(self, path: str):
        meta_path = os.path.join(path, 'metadata.json')
        if not os.path.exists(meta_path):
            return
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        self._task_order = meta['task_order']
        for layer in meta['layers']:
            layer_dir = os.path.join(path, layer.replace('.', '_'))
            if not os.path.exists(layer_dir):
                continue
            for task_id in self._task_order:
                fpath = os.path.join(layer_dir, f'{task_id}.pt')
                if os.path.exists(fpath):
                    data = torch.load(fpath, map_location=self._device)
                    self._storage[layer][task_id] = TaskMemory(
                        U=data['U'].to(self._device),
                        sigma=data['sigma'].to(self._device),
                        V=data['V'].to(self._device),
                        task_id=data['task_id']
                    )