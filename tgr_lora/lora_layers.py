import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

from .math_utils import pad_orthogonal_columns, route_gradient_B, route_gradient_A, SynthesizedSubspaces
from .config import TGRLoRAConfig

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._subspaces: Optional[SynthesizedSubspaces] = None
        self._boost = 0.5
        self._damp = 1.0
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def reset_for_probe(self, scale: float = 0.01):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=scale)
    
    def smart_initialize(self, subspaces: SynthesizedSubspaces, scale: float = 0.01):
        device, dtype = self.lora_A.device, self.lora_A.dtype
        
        has_u = subspaces.U_reuse is not None and subspaces.U_reuse.shape[1] > 0
        has_v = subspaces.V_reuse is not None and subspaces.V_reuse.shape[1] > 0
        
        if has_u:
            U = subspaces.U_reuse.to(device=device, dtype=dtype)
            B_init = U[:, :self.rank] if U.shape[1] >= self.rank else pad_orthogonal_columns(U, self.rank)
            self.lora_B.data.copy_(B_init * scale)
        else:
            nn.init.zeros_(self.lora_B)
        
        if has_v:
            V = subspaces.V_reuse.to(device=device, dtype=dtype)
            A_init = V[:, :self.rank].T if V.shape[1] >= self.rank else pad_orthogonal_columns(V, self.rank).T
            self.lora_A.data.copy_(A_init)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        if has_v and not has_u:
            nn.init.normal_(self.lora_B, mean=0.0, std=scale)
    
    def set_subspaces(self, subspaces: SynthesizedSubspaces, boost: float = 0.5, damp: float = 0.8):
        self._subspaces = subspaces
        self._boost = boost
        self._damp = damp
    
    def clear_subspaces(self):
        self._subspaces = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype != self.lora_A.dtype:
            x = x.to(self.lora_A.dtype)
        
        out = self.dropout(x)
        out = F.linear(F.linear(out, self.lora_A), self.lora_B) * self.scaling
        
        return out.to(orig_dtype) if out.dtype != orig_dtype else out
    
    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def apply_gradient_routing(self):
        if self._subspaces is None:
            return
        
        sub = self._subspaces
        if self.lora_B.grad is not None:
            self.lora_B.grad.copy_(route_gradient_B(
                self.lora_B.grad, sub.U_protect, sub.U_reuse,
                self._boost, self._damp
            ))
        
        if self.lora_A.grad is not None:
            self.lora_A.grad.copy_(route_gradient_A(
                self.lora_A.grad, sub.V_protect, sub.V_reuse,
                self._boost, self._damp
            ))


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.lora = LoRALayer(
            original.in_features, original.out_features, rank, alpha, dropout,
            device=original.weight.device, dtype=original.weight.dtype
        )
        for p in self.original.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)
    
    def merge_lora(self):
        with torch.no_grad():
            self.original.weight.add_(self.lora.get_delta_weight().to(self.original.weight.dtype))


class TGRLoRAModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: TGRLoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self._lora_layers: Dict[str, LoRALinear] = {}
        self._inject_lora()
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def _inject_lora(self):
        targets = set(self.config.target_modules)
        to_replace = []
        
        for name, module in self.base_model.named_modules():
            if name.split('.')[-1] in targets and isinstance(module, nn.Linear):
                to_replace.append(name)
        
        for name in to_replace:
            target = self.base_model.get_submodule(name)
            parent_name, short = name.rsplit('.', 1) if '.' in name else ('', name)
            parent = self.base_model.get_submodule(parent_name) if parent_name else self.base_model
            
            lora = LoRALinear(target, self.config.rank, self.config.alpha, self.config.dropout)
            setattr(parent, short, lora)
            self._lora_layers[name] = lora
        
        logger.info(f"Injected LoRA into {len(self._lora_layers)} layers")
    
    @property
    def lora_layer_names(self) -> List[str]:
        return list(self._lora_layers.keys())
    
    def get_lora_layer(self, name: str) -> Optional[LoRALinear]:
        return self._lora_layers.get(name)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
    
    def get_lora_params(self) -> List[nn.Parameter]:
        params = []
        for layer in self._lora_layers.values():
            params.extend([layer.lora.lora_A, layer.lora.lora_B])
        return params
    
    def get_probe_updates(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return {name: layer.lora.get_delta_weight().detach().clone()
                    for name, layer in self._lora_layers.items()}
    
    def reset_all_lora(self):
        for layer in self._lora_layers.values():
            layer.lora.reset_parameters()
    
    def reset_all_lora_for_probe(self, scale: float = 0.01):
        for layer in self._lora_layers.values():
            layer.lora.reset_for_probe(scale)
    
    def smart_initialize_all(self, subspaces: Dict[str, SynthesizedSubspaces], scale: float = 0.01):
        for name, layer in self._lora_layers.items():
            if name in subspaces:
                layer.lora.smart_initialize(subspaces[name], scale)
            else:
                layer.lora.reset_parameters()
    
    def set_subspaces_all(self, subspaces: Dict[str, SynthesizedSubspaces], boost: float = 0.5, damp: float = 0.8):
        for name, layer in self._lora_layers.items():
            if name in subspaces:
                layer.lora.set_subspaces(subspaces[name], boost, damp)
    
    def apply_gradient_routing_all(self):
        for layer in self._lora_layers.values():
            layer.lora.apply_gradient_routing()
    
    def clear_subspaces_all(self):
        for layer in self._lora_layers.values():
            layer.lora.clear_subspaces()
    
    def merge_all_lora(self):
        for layer in self._lora_layers.values():
            layer.merge_lora()
    
    def freeze_base_model(self):
        for p in self.base_model.parameters():
            p.requires_grad = False
        for layer in self._lora_layers.values():
            layer.lora.lora_A.requires_grad = True
            layer.lora.lora_B.requires_grad = True
        logger.info("Base model frozen, only LoRA trainable")
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")