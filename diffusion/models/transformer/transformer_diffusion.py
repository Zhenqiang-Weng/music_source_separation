"""
基于Transformer的Diffusion模型实现
"""

import torch
import torch.nn as nn
import math

from ..registry import register_diffusion_model


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入
    
    将时间步t编码为高维向量,使用Transformer中的位置编码方式
    """
    
    def __init__(self, dim, max_period=1000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: [B] 时间步,范围[0, 1]
            
        Returns:
            embedding: [B, dim] 时间嵌入
        """
        device = t.device
        half_dim = self.dim // 2
        
        # 计算频率
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        
        # 计算相位
        args = t[:, None].float() * freqs[None, :]
        
        # 正弦和余弦
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # 如果dim是奇数,pad一位
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


@register_diffusion_model('transformer')
class TransformerDiffusionModel(nn.Module):
    """
    基于Transformer的Diffusion模型
    
    适合序列数据(如音频波形)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)
        
        # 输入投影(假设输入是1D或2D)
        # 对于音频: [B, T] 或 [B, C, T]
        self.input_proj = nn.Linear(1, config.hidden_size)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_hidden_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(config.hidden_size, 1)
        
    def forward(self, x, t):
        """
        Args:
            x: [B, T] 或 [B, C, T] 输入波形
            t: [B] 时间步
            
        Returns:
            velocity: [B, T] 或 [B, C, T] 预测的速度场
        """
        orig_shape = x.shape
        
        # 处理多通道情况
        if x.ndim == 3:
            B, C, T = x.shape
            x = x.transpose(1, 2).reshape(B * C, T)  # [B*C, T]
            t = t.repeat_interleave(C)  # [B*C]
        else:
            B, T = x.shape
            C = 1
        
        # 输入投影: [B, T] -> [B, T, hidden_size]
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.input_proj(x)  # [B, T, hidden_size]
        
        # 时间嵌入: [B] -> [B, 1, hidden_size]
        t_embed = self.time_embed(t).unsqueeze(1)
        
        # 添加时间嵌入
        x = x + t_embed
        
        # Transformer
        x = self.transformer(x)  # [B, T, hidden_size]
        
        # 输出投影
        x = self.output_proj(x).squeeze(-1)  # [B, T]
        
        # 恢复原始shape
        if len(orig_shape) == 3:
            x = x.reshape(B, C, T).transpose(1, 2)  # [B, C, T]
        
        return x
