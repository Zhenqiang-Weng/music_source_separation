"""
简单的MLP Diffusion模型实现
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


@register_diffusion_model('simple_mlp')
class SimpleMLP(nn.Module):
    """
    简单的MLP模型(用于测试)
    
    输入: (batch, channels, time) 或 (batch, time)
    输出: 与输入相同shape
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.hidden_size)
        )
        
    def forward(self, x, t):
        """
        Args:
            x: [B, ...] 输入数据
            t: [B] 时间步
            
        Returns:
            velocity: [B, ...] 预测的速度场
        """
        # 保存原始shape
        orig_shape = x.shape
        batch_size = x.shape[0]
        
        # 展平
        x_flat = x.view(batch_size, -1)
        
        # 时间嵌入
        t_embed = self.time_embed(t)  # [B, hidden_size]
        
        # 全局池化作为输入特征
        if x_flat.shape[1] > self.config.hidden_size:
            # 如果输入维度太大,使用线性投影
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x_flat.shape[1], self.config.hidden_size).to(x.device)
            x_feat = self.input_proj(x_flat)
        else:
            # 否则pad到hidden_size
            x_feat = torch.nn.functional.pad(x_flat, (0, self.config.hidden_size - x_flat.shape[1]))
        
        # 拼接特征
        feat = torch.cat([x_feat, t_embed], dim=-1)  # [B, hidden_size * 2]
        
        # MLP处理
        out = self.mlp(feat)  # [B, hidden_size]
        
        # 投影回原始维度
        if out.shape[1] != x_flat.shape[1]:
            if not hasattr(self, 'output_proj'):
                self.output_proj = nn.Linear(out.shape[1], x_flat.shape[1]).to(x.device)
            out = self.output_proj(out)
        
        # 恢复原始shape
        out = out.view(orig_shape)
        
        return out
