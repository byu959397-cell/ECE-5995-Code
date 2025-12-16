import math
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import logging

from .config import TGRLoRAConfig, TaskConfig
from .memory_bank import MemoryBank
from .lora_layers import TGRLoRAModel
from .math_utils import SynthesizedSubspaces

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    task_id: str
    final_loss: float
    training_time: float
    similar_tasks: List[str]
    different_tasks: List[str]


class TGRLoRATrainer:
    def __init__(self, model: TGRLoRAModel, config: TGRLoRAConfig, memory_bank: MemoryBank, device: torch.device):
        self.model = model
        self.config = config
        self.memory_bank = memory_bank
        self.device = device
        
        self.model.to(device)
        self.memory_bank.set_device(device)
        self._current_subspaces: Optional[Dict[str, SynthesizedSubspaces]] = None
    
    def probe_task(self, dataloader: DataLoader, task_config: TaskConfig) -> Dict[str, SynthesizedSubspaces]:
        logger.info(f"Phase 1: Probing {task_config.task_id}")
        
        if self.memory_bank.num_tasks == 0 or task_config.skip_probing:
            logger.info("First task or skipped")
            return {name: SynthesizedSubspaces() for name in self.model.lora_layer_names}

        self.model.reset_all_lora_for_probe(scale=0.01)
        self.model.freeze_base_model()
        
        params = self.model.get_lora_params()
        optimizer = AdamW(params, lr=self.config.probe_lr)
        steps = task_config.custom_probe_steps or self.config.probe_steps
        
        self.model.train()
        data_iter = iter(dataloader)
        
        for step in tqdm(range(steps), desc="Probing"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            if batch.get('_skip_batch', False):
                continue
            
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            loss = self.model(**batch).loss
            
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            optimizer.step()

        # {layername: synthesizedsubspaces, ...}
        return self.memory_bank.synthesize_all_layers(self.model.get_probe_updates())

    # 一initialize layer by layer
    def initialize_for_task(self, subspaces: Dict[str, SynthesizedSubspaces], task_config: TaskConfig):
        logger.info(f"Phase 2: Initializing {task_config.task_id}")
        self.model.reset_all_lora()
        self.model.smart_initialize_all(subspaces, scale=0.01)
        self.model.set_subspaces_all(subspaces, self.config.boost_factor, self.config.damp_factor)
        self._current_subspaces = subspaces
    
    def train_task(self, dataloader: DataLoader, task_config: TaskConfig) -> float:
        logger.info(f"Phase 3: Training {task_config.task_id}")
        
        epochs = task_config.num_epochs or self.config.num_epochs
        lr = task_config.learning_rate or self.config.learning_rate
        accum = self.config.gradient_accumulation_steps
        
        params = self.model.get_lora_params()
        optimizer = AdamW(params, lr=lr, weight_decay=self.config.weight_decay)
        
        total_steps = math.ceil(len(dataloader) / accum) * epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if warmup_steps > 0:
            scheduler = SequentialLR(optimizer, [
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
            ], milestones=[warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        
        self.model.train()
        final_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss, n = 0.0, 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                if batch.get('_skip_batch', False):
                    continue
                
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                loss = self.model(**batch).loss / accum
                
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                
                if (i + 1) % accum == 0 or (i + 1) == len(dataloader):
                    # adjust the gradient layer by layer
                    self.model.apply_gradient_routing_all()
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                    
                    if all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accum
                n += 1
            
            final_loss = epoch_loss / max(n, 1)
            logger.info(f"Epoch {epoch+1} loss={final_loss:.4f}")
        
        return final_loss
    
    def consolidate_task(self, task_config: TaskConfig):
        logger.info(f"Phase 4: Consolidating {task_config.task_id}")
        
        for name in self.model.lora_layer_names:
            layer = self.model.get_lora_layer(name)
            self.memory_bank.add_memory(
                name, task_config.task_id,
                layer.lora.lora_B.data.detach(),
                layer.lora.lora_A.data.detach()
            )
        
        self.model.merge_all_lora()
        self.model.reset_all_lora()
        self.model.clear_subspaces_all()
        
        if self.config.memory_bank_path:
            self.memory_bank.save(self.config.memory_bank_path)
    
    def learn_task(self, task_id: str, train_dataloader: DataLoader, task_config: Optional[TaskConfig] = None) -> TaskResult:
        start = time.time()
        task_config = task_config or TaskConfig(task_id=task_id)
        
        logger.info(f"\n{'='*50}\nLearning: {task_config.task_name}\n{'='*50}")
        
        subspaces = self.probe_task(train_dataloader, task_config)
        self.initialize_for_task(subspaces, task_config)
        loss = self.train_task(train_dataloader, task_config)
        self.consolidate_task(task_config)
        
        similar, different = [], []
        if self._current_subspaces:
            for sub in self._current_subspaces.values():
                similar.extend(sub.similar_tasks)
                different.extend(sub.different_tasks)
        
        self._current_subspaces = None
        
        return TaskResult(
            task_id=task_id, final_loss=loss,
            training_time=time.time() - start,
            similar_tasks=list(set(similar)),
            different_tasks=list(set(different))
        )