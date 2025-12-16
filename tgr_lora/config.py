from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TGRLoRAConfig:
    # LoRA
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.05
    
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    probe_steps: int = 50
    probe_lr: float = 5e-5
    similarity_threshold_in: float = 0.35
    similarity_threshold_out: float = 0.35
    boost_factor: float = 0.5
    damp_factor: float = 0.8
    
    learning_rate: float = 5e-5
    num_epochs: int = 3
    
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    max_seq_length: int = 1024
    memory_bank_path: Optional[str] = None
    
    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class TaskConfig:
    task_id: str
    task_name: Optional[str] = None
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    custom_probe_steps: Optional[int] = None
    skip_probing: bool = False
    
    def __post_init__(self):
        if self.task_name is None:
            self.task_name = self.task_id