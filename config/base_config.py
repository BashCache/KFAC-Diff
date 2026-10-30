from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class SparsityType(Enum):
    STRUCTURED    = "structured"
    SEMI_STRUCTURED = "semi_structured" 

class SparsityScope(Enum):
    GLOBAL           = "global"
    LOCAL            = "local"
    LAYERWISE_CUSTOM = "layerwise_custom"

@dataclass
class CalibrationConfig:
    dataset    : str = "c4"
    num_samples: int = 128
    seq_len    : int = 2048
    batch_size : int = 1
    seed       : int = 42

@dataclass
class SparsityConfig:
    sparsity_type     : SparsityType  = SparsityType.STRUCTURED
    scope             : SparsityScope = SparsityScope.GLOBAL
    target_sparsity   : float         = 0.5
    layer_sparsity_map: Dict[str, float] = field(default_factory=dict)
    structured_dim    : int           = 0   

@dataclass
class PruningConfig:
    # Required
    method              : str = "wanda"
    model_name_or_path  : str = ""

    # Sub-configs
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sparsity   : SparsityConfig    = field(default_factory=SparsityConfig)

    # Layer selection
    layers_to_prune: List[str] = field(default_factory=list) 
    layers_to_skip : List[str] = field(default_factory=list)

    # Method-specific pass-through kwargs
    method_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Runtime
    device    : str  = "cuda"
    dtype     : str  = "float16"
    seed      : int  = 42
    output_dir: str  = "./pruned_model"
    save_model: bool = True
    save_sparse: bool = False