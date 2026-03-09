"""
Compatibility shim for legacy imports.

This module re-exports the lightweight SampleLocationAndConditionalFlow
from the new `diffusion.utils` package so older imports continue to work.
"""
from inspect import isfunction
from typing import Callable, List, Optional, Sequence, TypeVar, Union, Dict, Tuple
from typing_extensions import TypeGuard
import collections.abc
from itertools import repeat


T = TypeVar("T")

class SampleLocationAndConditionalFlow:
    """
    简化的采样辅助类
    
    为了保持向后兼容性，提供静态方法接口
    """

    @staticmethod
    def run(matcher, x0, x1, t=None):
        """
        从ConditionalFlowMatcher采样xt和ut
        
        使用示例:
            fm = ConditionalFlowMatcher(sigma=0.1)
            noise = torch.randn(16, 32)
            x_real = torch.randn(16, 32)
            
            t, xt, ut = SampleLocationAndConditionalFlow.run(
                fm, x0=noise, x1=x_real
            )
        
        Parameters
        ----------
        matcher : ConditionalFlowMatcher
            Flow matcher实例
        x0, x1 : Tensor
            源和目标样本批次
        t : Tensor or None
            时间步; 如果为None，从Uniform(0,1)采样

        Returns
        -------
        t : Tensor, shape (bs,)
            采样的时间步
        xt : Tensor, shape (bs, *dim)
            中间状态
        ut : Tensor, shape (bs, *dim)
            条件流场
        """
        return matcher.sample_location_and_conditional_flow(x0, x1, t)
    
