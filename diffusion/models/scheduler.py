"""
学习率调度器 (Learning Rate Schedulers)

包含:
1. NoamScheduler: Noam学习率调度器（Transformer论文中使用的调度器）
2. WarmupScheduler: 简单的预热调度器
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class NoamScheduler(_LRScheduler):
    """
    Noam学习率调度器
    
    实现了Transformer论文"Attention is All You Need"中提出的学习率调度策略：
    lr = init_scale * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    特点：
    - 在预热阶段（warmup）线性增长
    - 预热后按步数的平方根倒数衰减
    - 适合深度学习模型，特别是Transformer架构
    
    Args:
        optimizer: PyTorch优化器
        n_warmup: 预热步数，通常设置为4000-8000
        init_scale: 初始缩放因子，用于调整学习率范围
        last_epoch: 上次训练的epoch，用于恢复训练
    
    使用示例:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = NoamScheduler(optimizer, n_warmup=4000, init_scale=0.5)
        
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()  # 每步更新学习率
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        n_warmup: int = 4000,
        init_scale: float = 1.0,
        last_epoch: int = -1
    ):
        """
        初始化Noam调度器
        
        Args:
            optimizer: PyTorch优化器
            n_warmup: 预热步数
            init_scale: 初始缩放因子
            last_epoch: 上次训练的epoch
        """
        self.n_warmup = n_warmup
        self.init_scale = init_scale
        self.n_steps = 0  # 当前步数
        
        super(NoamScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        计算当前学习率
        
        公式: lr = init_scale * min(step^(-0.5), step * n_warmup^(-1.5))
        
        Returns:
            学习率列表（每个参数组一个）
        """
        self.n_steps += 1
        
        # 计算缩放因子
        if self.n_steps < self.n_warmup:
            # 预热阶段：线性增长
            scale = self.n_steps / self.n_warmup
        else:
            # 衰减阶段：平方根倒数衰减
            scale = math.pow(self.n_warmup / self.n_steps, 0.5)
        
        # 应用初始缩放
        scale = self.init_scale * scale
        
        # 返回每个参数组的学习率
        return [base_lr * scale for base_lr in self.base_lrs]
    
    def state_dict(self):
        """保存调度器状态"""
        state = {
            'n_warmup': self.n_warmup,
            'init_scale': self.init_scale,
            'n_steps': self.n_steps,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }
        return state
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.n_warmup = state_dict['n_warmup']
        self.init_scale = state_dict['init_scale']
        self.n_steps = state_dict['n_steps']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']


class WarmupScheduler(_LRScheduler):
    """
    简单的预热调度器
    
    在指定的预热步数内线性增加学习率，之后保持不变或可选衰减
    
    Args:
        optimizer: PyTorch优化器
        n_warmup: 预热步数
        warmup_start_lr: 预热起始学习率（相对于base_lr的比例）
        decay_after_warmup: 预热后是否衰减
        decay_rate: 衰减率（如果decay_after_warmup=True）
        last_epoch: 上次训练的epoch
    
    使用示例:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(
            optimizer, 
            n_warmup=2000, 
            warmup_start_lr=0.1,
            decay_after_warmup=False
        )
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_warmup: int = 2000,
        warmup_start_lr: float = 0.1,
        decay_after_warmup: bool = False,
        decay_rate: float = 0.9999,
        last_epoch: int = -1
    ):
        """
        初始化预热调度器
        
        Args:
            optimizer: PyTorch优化器
            n_warmup: 预热步数
            warmup_start_lr: 预热起始学习率（相对比例）
            decay_after_warmup: 预热后是否衰减
            decay_rate: 每步的衰减率
            last_epoch: 上次训练的epoch
        """
        self.n_warmup = n_warmup
        self.warmup_start_lr = warmup_start_lr
        self.decay_after_warmup = decay_after_warmup
        self.decay_rate = decay_rate
        self.n_steps = 0
        
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        计算当前学习率
        
        Returns:
            学习率列表
        """
        self.n_steps += 1
        
        if self.n_steps <= self.n_warmup:
            # 预热阶段：从warmup_start_lr线性增加到1.0
            scale = self.warmup_start_lr + (1.0 - self.warmup_start_lr) * (self.n_steps / self.n_warmup)
        else:
            # 预热后
            if self.decay_after_warmup:
                # 指数衰减
                steps_after_warmup = self.n_steps - self.n_warmup
                scale = math.pow(self.decay_rate, steps_after_warmup)
            else:
                # 保持不变
                scale = 1.0
        
        return [base_lr * scale for base_lr in self.base_lrs]
    
    def state_dict(self):
        """保存调度器状态"""
        state = {
            'n_warmup': self.n_warmup,
            'warmup_start_lr': self.warmup_start_lr,
            'decay_after_warmup': self.decay_after_warmup,
            'decay_rate': self.decay_rate,
            'n_steps': self.n_steps,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }
        return state
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.n_warmup = state_dict['n_warmup']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.decay_after_warmup = state_dict['decay_after_warmup']
        self.decay_rate = state_dict['decay_rate']
        self.n_steps = state_dict['n_steps']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    带预热的余弦退火调度器
    
    结合了预热和余弦退火，广泛用于现代深度学习训练
    
    Args:
        optimizer: PyTorch优化器
        n_warmup: 预热步数
        T_max: 余弦周期的最大步数
        eta_min: 最小学习率
        last_epoch: 上次训练的epoch
    
    使用示例:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, 
            n_warmup=2000,
            T_max=100000,
            eta_min=1e-6
        )
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_warmup: int = 2000,
        T_max: int = 100000,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        """
        初始化带预热的余弦退火调度器
        
        Args:
            optimizer: PyTorch优化器
            n_warmup: 预热步数
            T_max: 余弦周期的最大步数（从预热结束后开始计算）
            eta_min: 最小学习率
            last_epoch: 上次训练的epoch
        """
        self.n_warmup = n_warmup
        self.T_max = T_max
        self.eta_min = eta_min
        self.n_steps = 0
        
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        计算当前学习率
        
        预热阶段使用线性增长，之后使用余弦退火
        
        Returns:
            学习率列表
        """
        self.n_steps += 1
        
        if self.n_steps <= self.n_warmup:
            # 预热阶段
            scale = self.n_steps / self.n_warmup
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            steps_after_warmup = self.n_steps - self.n_warmup
            progress = min(steps_after_warmup / self.T_max, 1.0)
            
            # 余弦退火公式
            cos_scale = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cos_scale 
                for base_lr in self.base_lrs
            ]
    
    def state_dict(self):
        """保存调度器状态"""
        state = {
            'n_warmup': self.n_warmup,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'n_steps': self.n_steps,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }
        return state
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.n_warmup = state_dict['n_warmup']
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.n_steps = state_dict['n_steps']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']


# 便利函数
def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    **kwargs
):
    """
    根据名称获取调度器
    
    Args:
        name: 调度器名称 ('noam', 'warmup', 'cosine')
        optimizer: PyTorch优化器
        **kwargs: 调度器参数
    
    Returns:
        调度器实例
    
    使用示例:
        scheduler = get_scheduler('noam', optimizer, n_warmup=4000)
    """
    name = name.lower()
    
    if name == 'noam':
        return NoamScheduler(optimizer, **kwargs)
    elif name == 'warmup':
        return WarmupScheduler(optimizer, **kwargs)
    elif name == 'cosine':
        return CosineAnnealingWarmupScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"未知的调度器名称: {name}")
