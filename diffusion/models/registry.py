"""
模型注册表与装饰器

集中管理 diffusion 模型的注册与查询。
"""
import os
import importlib

DIFFUSION_MODEL_REGISTRY = {}


def register_diffusion_model(name):
    """
    装饰器：注册 diffusion 模型类

    使用方法:
        @register_diffusion_model('transformer')
        class TransformerDiffusionModel(nn.Module):
            ...
    """
    def decorator(cls):
        DIFFUSION_MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_diffusion_model(name):
    """
    根据名称实例化 diffusion 模型
    """
    name = name.lower()
    if name not in DIFFUSION_MODEL_REGISTRY:
        available = ', '.join(DIFFUSION_MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知的模型类型: {name}\n可用的模型: {available}"
        )
    model_class = DIFFUSION_MODEL_REGISTRY[name]
    return model_class()


def list_diffusion_models():
    return list(DIFFUSION_MODEL_REGISTRY.keys())


# 自动导入并注册所有模型
_current_dir = os.path.dirname(__file__)
for _filename in os.listdir(_current_dir):
    _filepath = os.path.join(_current_dir, _filename)
    
    # 导入模块文件夹
    if os.path.isdir(_filepath) and not _filename.startswith('_'):
        _init_file = os.path.join(_filepath, '__init__.py')
        if os.path.exists(_init_file):
            importlib.import_module(f'diffusion.models.{_filename}')



