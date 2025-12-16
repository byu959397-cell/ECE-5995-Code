# __init__.py
from .config import TGRLoRAConfig, TaskConfig
from .math_utils import (
    TaskMemory, SynthesizedSubspaces,
    truncated_svd, stable_qr_orthogonalize,
    bilateral_similarity,
    route_gradient_B, route_gradient_A,
    pad_orthogonal_columns
)
from .memory_bank import MemoryBank
from .lora_layers import LoRALayer, LoRALinear, TGRLoRAModel
from .trainer import TGRLoRATrainer, TaskResult
from .evaluation import TRACEEvaluator, TaskMetrics, ContinualMetrics, MetricsCalculator
from .trace_dataset import TRACEDataset, TRACEDataModule, TRACE_TASKS

__version__ = "1.0.0"

__all__ = [
    "TGRLoRAConfig", "TaskConfig",
    "TaskMemory", "SynthesizedSubspaces",
    "truncated_svd", "stable_qr_orthogonalize", "bilateral_similarity",
    "route_gradient_B", "route_gradient_A", "pad_orthogonal_columns",
    "MemoryBank", "LoRALayer", "LoRALinear", "TGRLoRAModel",
    "TGRLoRATrainer", "TaskResult",
    "TRACEEvaluator", "TaskMetrics", "ContinualMetrics", "MetricsCalculator",
    "TRACEDataset", "TRACEDataModule", "TRACE_TASKS",
]