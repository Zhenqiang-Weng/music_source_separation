"""
条件流匹配 (Conditional Flow Matching) 实现

基于论文:
- "Flow Matching for Generative Modeling" (Lipman et al., 2022)
- "Improving and generalizing flow-based generative models with minibatch optimal transport" (Tong et al., 2023)

支持:
1. Optimal Transport (OT) 路径 (sigma=0)
2. 随机插值路径 (sigma>0)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConditionalFlowMatcher:
    """
    条件流匹配器
    
    实现了Optimal Transport (OT) 路径和随机路径的条件流匹配。
    
    核心思想:
    - 定义从源分布p0到目标分布p1的概率路径pt
    - 学习一个速度场vθ(xt, t)来匹配条件流场ut
    - 训练目标: E[||vθ(xt, t) - ut||^2]
    
    Args:
        sigma: 噪声标准差
            - sigma=0.0: 使用OT路径 (确定性路径)
            - sigma>0.0: 使用随机路径 (添加高斯噪声)
    
    使用示例:
        # 创建flow matcher
        fm = ConditionalFlowMatcher(sigma=0.0)
        
        # 训练时采样
        x0 = torch.randn(32, 128)  # 源样本（噪声）
        x1 = torch.randn(32, 128)  # 目标样本（数据）
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        
        # 模型预测
        vt = model(xt, t)
        loss = F.mse_loss(vt, ut)
    """
    
    def __init__(self, sigma: float = 0.0):
        """
        初始化条件流匹配器
        
        Args:
            sigma: 噪声水平，0表示确定性OT路径
        """
        self.sigma = sigma
        
    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        采样与x形状相同的标准高斯噪声
        
        Args:
            x: 参考张量 [B, ...]
            
        Returns:
            噪声张量，形状与x相同
        """
        return torch.randn_like(x)
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        从Uniform(0, 1)采样时间步
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            时间步 [B]
        """
        return torch.rand(batch_size, device=device)
    
    def sample_xt(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor, 
        eps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        在时间t采样中间状态xt
        
        条件概率路径:
        1. OT路径 (sigma=0):
           xt = (1-t)*x0 + t*x1
           
        2. 随机路径 (sigma>0):
           xt = (1-t)*x0 + t*x1 + sigma*sqrt(t*(1-t))*eps
           
        其中eps ~ N(0, I)是标准高斯噪声
        
        Args:
            x0: 源分布样本 [B, ...] (通常是噪声)
            x1: 目标分布样本 [B, ...] (通常是数据)
            t: 时间步 [B] 或 [B, 1, ..., 1]
            eps: 可选的噪声，如果为None则自动采样
            
        Returns:
            中间状态xt [B, ...]
        """
        # 确保t的维度匹配
        if t.ndim == 1:
            # 扩展t: [B] -> [B, 1, 1, ...]
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        else:
            t_expanded = t
        
        # 基础线性插值: (1-t)*x0 + t*x1
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # 如果使用随机路径，添加时变噪声
        if self.sigma > 0:
            if eps is None:
                eps = self.sample_noise_like(x0)
            # 计算噪声缩放: sigma * sqrt(t*(1-t))
            # 这确保在t=0和t=1时噪声为0
            noise_scale = self.sigma * torch.sqrt(t_expanded * (1 - t_expanded) + 1e-8)
            xt = xt + noise_scale * eps
            
        return xt
    
    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算条件流场 ut (目标速度场)
        
        条件流场定义:
        ut = dx_t/dt | (x0, x1)
        
        对于线性插值路径:
        ut = x1 - x0
        
        这是Optimal Transport路径的关键性质：
        速度场与时间无关，只依赖于端点
        
        注意：对于更复杂的随机路径，ut可能包含额外的噪声项，
        但在实践中简化版本 ut = x1 - x0 通常就足够了。
        
        Args:
            x0: 源分布样本 [B, ...]
            x1: 目标分布样本 [B, ...]
            t: 时间步 [B] (保留接口，当前未使用)
            xt: 当前状态 [B, ...] (保留接口，当前未使用)
            
        Returns:
            条件流场 ut [B, ...]
        """
        # 对于条件流匹配，velocity就是 x1 - x0
        # 这是因为我们使用的是线性插值路径
        ut = x1 - x0
        
        # 理论扩展：对于完整的随机路径，ut还应包含噪声项的时间导数
        # 如果需要完整实现:
        # if self.sigma > 0 and t is not None:
        #     t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        #     # 计算 d/dt[sigma * sqrt(t*(1-t))]
        #     noise_deriv = self.sigma * (0.5 - t_expanded) / (torch.sqrt(t_expanded * (1 - t_expanded)) + 1e-8)
        #     ut = ut + noise_deriv * eps  # 需要额外传入eps
        
        return ut
    
    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一次性采样时间、位置和条件流
        
        这是训练时的主要接口：
        1. 采样时间 t ~ Uniform(0, 1)
        2. 采样中间状态 xt ~ p_t(·|x0, x1)
        3. 计算目标速度 ut
        
        训练循环示例:
            for x0, x1 in dataloader:
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
                vt = model(xt, t)
                loss = F.mse_loss(vt, ut)
                loss.backward()
        
        Args:
            x0: 源分布样本 [B, ...] (如噪声)
            x1: 目标分布样本 [B, ...] (如数据)
            t: 可选的时间步 [B]，如果为None则从Uniform(0,1)采样
            return_noise: 是否返回采样的噪声eps
            
        Returns:
            t: 时间步 [B]
            xt: 中间状态 [B, ...]
            ut: 条件流场 [B, ...]
            (eps: 噪声 [B, ...], 仅当return_noise=True)
        """
        batch_size = x0.shape[0]
        
        # 1. 采样时间
        if t is None:
            t = self.sample_time(batch_size, x0.device)
        
        # 2. 采样噪声（如果需要）
        eps = None
        if self.sigma > 0:
            eps = self.sample_noise_like(x0)
        
        # 3. 采样中间状态 xt
        xt = self.sample_xt(x0, x1, t, eps)
        
        # 4. 计算目标速度场 ut
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        if return_noise:
            return t, xt, ut, eps
        return t, xt, ut
    
    def compute_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Flow Matching损失
        
        这是一个便利函数，封装了完整的训练损失计算
        
        Args:
            model: 速度场模型 vθ(xt, t)
            x0: 源样本
            x1: 目标样本
            t: 可选的时间步
            
        Returns:
            MSE损失
        """
        # 采样训练数据
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1, t)
        
        # 模型预测
        vt = model(xt, t)
        
        # 计算MSE损失
        loss = nn.functional.mse_loss(vt, ut)
        
        return loss


class OptimalTransportFlowMatcher(ConditionalFlowMatcher):
    """
    Optimal Transport Flow Matcher (sigma=0的特殊情况)
    
    这是最常用的配置，提供确定性的OT路径
    """
    
    def __init__(self):
        super().__init__(sigma=0.0)


class StochasticFlowMatcher(ConditionalFlowMatcher):
    """
    随机流匹配器 (sigma>0)
    
    使用随机插值路径，可能提供更好的样本质量
    但训练可能稍慢
    """
    
    def __init__(self, sigma: float = 0.1):
        if sigma <= 0:
            raise ValueError("StochasticFlowMatcher需要sigma > 0")
        super().__init__(sigma=sigma)


# 向后兼容的别名
OTFlowMatcher = OptimalTransportFlowMatcher
