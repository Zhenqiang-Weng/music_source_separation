"""
Diffusion模型注册表与默认模型实现
模型类由 `diffusion.models.registry` 管理，具体模型实现放在各自文件中。
"""

from .registry import register_diffusion_model, get_diffusion_model, list_diffusion_models


# 便利函数：创建默认模型
def create_default_model(config):
    """创建默认的diffusion模型"""
    model_type = getattr(config, 'model_type', 'transformer')
    return get_diffusion_model(model_type, config)
