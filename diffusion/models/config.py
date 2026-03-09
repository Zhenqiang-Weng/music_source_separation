"""
DiffusionConfig dataclass - central config used by models and wrappers

Place common hyperparameters here so models and training scripts share the same
configuration shape.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffusionConfig:
    """Diffusion配置类"""
    # 模型参数
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden_size: int = 1024
    dropout: float = 0.1
    
    # Flow Matching参数
    sigma: float = 0.0  # 条件流匹配的噪声标准差，0表示OT路径
    
    # 训练参数
    n_train_start: int = 0  # 延迟启动步数
    diffusion_loss_weight: float = 1.0  # diffusion损失权重
    
    # 优化器参数
    lr: float = 0.0001
    betas: Optional[list] = None  # [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0.0
    
    # 调度器参数
    n_warmup: int = 4000
    init_scale: float = 0.5
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1

    def __post_init__(self):
        if self.betas is None:
            self.betas = [0.9, 0.999]
