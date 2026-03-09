"""
模板 diffuser 实现示例

把新的模型实现放到单独的文件,并用 @register_diffusion_model 注册。
"""

import torch
import torch.nn as nn

from ..registry import register_diffusion_model


@register_diffusion_model('template')
class TemplateDiffuser(nn.Module):
    """最小示例模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, x, t):
        # 适配输入到 [B, T, 1] -> flatten
        orig_shape = x.shape
        x_in = x.unsqueeze(-1) if x.ndim == 2 else x
        out = self.net(x_in)
        return out.squeeze(-1)
