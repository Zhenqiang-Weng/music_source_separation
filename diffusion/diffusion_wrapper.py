"""
Diffusion包装类 - 支持Accelerate并行训练
可插入式diffusion模块，支持条件流匹配训练

使用示例:
    from diffusion_wrapper import DiffusionWrapper
    
    # 初始化
    diff_wrapper = DiffusionWrapper(
        config=diff_config,
        accelerator=accelerator  # 从accelerate导入的accelerator对象
    )
    
    # 训练步骤（双波形输入）
    losses = diff_wrapper.train_step_dual(wave_source, wave_target, current_step)
    
    # 训练步骤（单波形输入）
    losses = diff_wrapper.train_step_single(wave, current_step)
    
    # 保存checkpoint
    diff_wrapper.save_checkpoint("path/to/checkpoint.pth")
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from dataclasses import dataclass

from diffusion.models.utils import SampleLocationAndConditionalFlow


@dataclass
class DiffusionConfig:
    """Diffusion配置类 - 仅包含训练和优化器配置"""
    # 模型选择
    model_type: str = 'dit'  # 模型类型: 'transformer', 'unet', 'dit' 等
    
    # Flow Matching参数
    sigma: float = 0.0  # 条件流匹配的噪声标准差，0表示OT路径
    
    # 训练参数
    n_train_start: int = 100*1000  # 延迟启动步数
    diffusion_loss_weight: float = 1.0  # diffusion损失权重
    
    # 优化器参数
    optimizer_type: str = 'adam'  # 优化器类型: 'adam', 'adamw', 'sgd'
    lr: float = 9e-5
    betas: list = None  # [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0.0
    
    # 调度器参数
    scheduler_type: str = 'noam'  # 调度器类型: 'noam', 'warmup', 'cosine', 'constant'
    n_warmup: int = 4000
    init_scale: float = 0.5
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1
    
    # Cosine调度器专用参数
    T_max: int = 10  # 余弦周期的最大步数
    eta_min: float = 0.0  # 最小学习率
    
    # STFT参数
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: str = 'hann'
    center: bool = True
    normalized: bool = False
    use_stft: bool = True  # 是否使用STFT（DiT模型需要）
    
    # 推理参数
    chunk_size_seconds: float = 8.0  # 分块推理的块大小（秒）
    sample_rate: int = 44100  # 采样率
    num_overlap: int = 2  # 重叠倍数
    inference_batch_size: int = 1  # 推理批次大小
    
    def __post_init__(self):
        if self.betas is None:
            self.betas = [0.9, 0.999]
        if self.model_type in ["dit"]:
            self.use_stft = True
        else:
            self.use_stft = False

class DiffusionWrapper:
    """
    Diffusion训练包装类
    
    封装了diffusion model的初始化、训练、保存等功能，
    并支持accelerate并行训练
    """
    
    def __init__(
        self, 
        config: Union[DiffusionConfig, dict],
        accelerator=None,
        device: Optional[torch.device] = None
    ):
        """
        初始化Diffusion包装类
        
        Args:
            config: 配置对象或字典
            accelerator: accelerate的Accelerator对象，用于并行训练
            device: 设备，如果提供accelerator则忽略此参数
        """
        # 配置处理
        if isinstance(config, dict):
            self.config = DiffusionConfig(**config)
        else:
            self.config = config
            
        self.accelerator = accelerator
        self.device = device if accelerator is None else None
        
        # 初始化组件
        self._init_diffusion_model()
        self._init_flow_matcher()
        self._init_stft()
        self._init_optimizer()
        
        # 如果使用accelerate，准备模型和优化器
        if self.accelerator is not None:
            self.model, self.optimizer, self.scheduler = \
                self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        elif self.device is not None:
            self.model = self.model.to(self.device)
    
    def _init_stft(self):
        """初始化STFT变换"""
        if not getattr(self.config, 'use_stft', True):
            self.stft_fn = None
            self.istft_fn = None
            return
        
        # 创建窗函数
        if self.config.window == 'hann':
            window = torch.hann_window(self.config.win_length)
        elif self.config.window == 'hamming':
            window = torch.hamming_window(self.config.win_length)
        elif self.config.window == 'blackman':
            window = torch.blackman_window(self.config.win_length)
        else:
            window = None
        
        # 保存窗函数
        self.window = window
        
        # STFT参数
        self.stft_params = {
            'n_fft': self.config.n_fft,
            'hop_length': self.config.hop_length,
            'win_length': self.config.win_length,
            'window': window,
            'center': self.config.center,
            'normalized': self.config.normalized,
            'return_complex': True
        }
        
        # iSTFT参数
        self.istft_params = {
            'n_fft': self.config.n_fft,
            'hop_length': self.config.hop_length,
            'win_length': self.config.win_length,
            'window': window,
            'center': self.config.center,
            'normalized': self.config.normalized
        }
    
    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        """
        执行STFT变换
        
        Args:
            wave: [B, C, T] 其中C为声道数（如2为立体声）
            
        Returns:
            spec: [B*C, 2, F, T] 其中2表示实部和虚部，F = n_fft // 2 + 1
        """
        if not getattr(self.config, 'use_stft', True):
            return wave
        
        # 确保窗函数在正确的设备上
        if self.window is not None:
            self.window = self.window.to(wave.device)
            self.stft_params['window'] = self.window
        
        # 处理输入维度
        if wave.ndim == 2:
            wave = wave.unsqueeze(1)  # [B, T] -> [B, 1, T]
        
        B, C, T = wave.shape
        
        # 对每个通道分别做STFT，然后折叠到batch维度
        specs = []
        for c in range(C):
            spec = torch.stft(wave[:, c, :], **self.stft_params)  # [B, F, T_spec] 复数
            specs.append(spec)
        
        # 堆叠所有声道: [B, C, F, T_spec]
        spec = torch.stack(specs, dim=1)
        
        # 折叠声道到batch维度: [B*C, F, T_spec]
        B, C, F, T_spec = spec.shape
        spec = spec.reshape(B * C, F, T_spec)
        
        # 分离实部和虚部: [B*C, 2, F, T_spec]
        spec_real = spec.real.unsqueeze(1)  # [B*C, 1, F, T_spec]
        spec_imag = spec.imag.unsqueeze(1)  # [B*C, 1, F, T_spec]
        spec_out = torch.cat([spec_real, spec_imag], dim=1)  # [B*C, 2, F, T_spec]
        
        return spec_out
    
    def istft(self, spec: torch.Tensor, length: Optional[int] = None, num_channels: int = 2) -> torch.Tensor:
        """
        执行iSTFT变换
        
        Args:
            spec: [B*C, 2, F, T] 其中2表示实部和虚部
            length: 可选的输出长度
            num_channels: 原始声道数（默认为2，即立体声）
            
        Returns:
            wave: [B, C, T]
        """
        if not getattr(self.config, 'use_stft', True):
            return spec
        
        # 确保窗函数在正确的设备上
        if self.window is not None:
            self.window = self.window.to(spec.device)
            self.istft_params['window'] = self.window
        
        BC, two, F, T_spec = spec.shape
        assert two == 2, f"Expected 2 channels (real/imag), got {two}"
        
        # 分离实部和虚部
        spec_real = spec[:, 0, :, :]  # [B*C, F, T_spec]
        spec_imag = spec[:, 1, :, :]  # [B*C, F, T_spec]
        
        # 合并为复数: [B*C, F, T_spec]
        spec_complex = torch.complex(spec_real, spec_imag)
        
        # 执行iSTFT: [B*C, T]
        wave = torch.istft(spec_complex, length=length, **self.istft_params)
        
        # 恢复声道维度: [B, C, T]
        B = BC // num_channels
        wave = wave.reshape(B, num_channels, -1)
        
        return wave
            
    def _init_diffusion_model(self):
        """初始化diffusion模型"""
        try:
            # 尝试导入模型注册表
            try:
                from diffusion.models.registry import get_diffusion_model
            except ImportError:
                try:
                    from .models.registry import get_diffusion_model
                except ImportError:
                    # 添加diffusion目录到路径
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.insert(0, current_dir)
                    from models.registry import get_diffusion_model
        except ImportError:
            raise ImportError("无法导入get_diffusion_model函数，请确保models.diffusion_model模块存在")
        
        # 根据配置中的model_type获取模型
        model_type = getattr(self.config, 'model_type', 'transformer')
        self.model = get_diffusion_model(model_type)
        
    def _init_flow_matcher(self):
        """初始化Flow Matcher"""
        try:
            # 尝试导入ConditionalFlowMatcher
            try:
                from diffusion.models.flow_matching import ConditionalFlowMatcher
            except ImportError:
                try:
                    from .models.flow_matching import ConditionalFlowMatcher
                except ImportError:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.insert(0, current_dir)
                    from models.flow_matching import ConditionalFlowMatcher
        except ImportError:
            raise ImportError("无法导入ConditionalFlowMatcher类，请确保models.flow_matching模块存在")
            
        self.flow_matcher = ConditionalFlowMatcher(sigma=self.config.sigma)
        
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        # 1. 创建优化器
        optimizer_type = getattr(self.config, 'optimizer_type', 'adamw').lower()
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")
        
        # 2. 创建学习率调度器
        scheduler_type = getattr(self.config, 'scheduler_type', 'noam').lower()
        
        if scheduler_type == 'noam':
            # 尝试导入NoamScheduler
            try:
                try:
                    from discriminator.models.duration import NoamScheduler
                except ImportError:
                    from diffusion.models.scheduler import NoamScheduler
                
                self.scheduler = NoamScheduler(
                    self.optimizer,
                    n_warmup=self.config.n_warmup,
                    init_scale=self.config.init_scale
                )
            except ImportError:
                # 回退到简单调度器
                print("警告: NoamScheduler不可用，使用简单预热调度器")
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / self.config.n_warmup)
                )
                
        elif scheduler_type == 'warmup':
            try:
                from diffusion.models.scheduler import WarmupScheduler
                self.scheduler = WarmupScheduler(
                    self.optimizer,
                    n_warmup=self.config.n_warmup
                )
            except ImportError:
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / self.config.n_warmup)
                )
                
        elif scheduler_type == 'cosine':
            try:
                from diffusion.models.scheduler import CosineAnnealingWarmupScheduler
                self.scheduler = CosineAnnealingWarmupScheduler(
                    self.optimizer,
                    n_warmup=self.config.n_warmup,
                    T_max=getattr(self.config, 'T_max', 100000),
                    eta_min=getattr(self.config, 'eta_min', 0.0)
                )
            except ImportError:
                # 使用PyTorch内置的余弦退火
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=getattr(self.config, 'T_max', 100000),
                    eta_min=getattr(self.config, 'eta_min', 0.0)
                )
                
        elif scheduler_type == 'constant':
            # 常量学习率（无调度）
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0
            )
        else:
            raise ValueError(f"未知的调度器类型: {scheduler_type}")
        
    def train_step_dual(
        self,
        wave_source: torch.Tensor,
        wave_target: torch.Tensor,
        current_step: int,
        t: Optional[torch.Tensor] = None,
        train_steps: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步diffusion训练（双波形输入）
        
        Args:
            wave_source: 源波形 [B, C, T]
            wave_target: 目标波形 [B, C, T]
            current_step: 当前训练步数
            t: 可选的时间步，如果为None则从T_max中均匀采样train_steps个时间点
            train_steps: 从T_max中采样的时间点数量
            
        Returns:
            包含各种损失的字典
        """
        losses = {}
        
        # 检查是否到了开始训练的步数
        if current_step < self.config.n_train_start:
            return {
                'diffusion_loss': torch.tensor(0.0, device=wave_source.device),
                'flow_matching_loss': torch.tensor(0.0, device=wave_source.device)
            }
        
        # 确保波形shape匹配
        assert wave_source.shape == wave_target.shape, "源波形和目标波形shape必须相同"
        
        # detach避免梯度传播
        wave_source = wave_source.detach()
        wave_target = wave_target.detach()
        
        # 保存原始波形信息
        original_length = wave_source.shape[-1]
        num_channels = wave_source.shape[1] if wave_source.ndim == 3 else 1
        
        # 如果使用STFT，转换为频谱图
        if getattr(self.config, 'use_stft', True):
            wave_source = self.stft(wave_source)  # [B*C, 2, F, T_spec]
            wave_target = self.stft(wave_target)  # [B*C, 2, F, T_spec]
        
        # 累积总损失
        total_flow_matching_loss = 0.0
        
        # 从T_max中采样train_steps个时间点
        if t is None:
            T_max = getattr(self.config, 'T_max', 10)
            actual_train_steps = min(train_steps, T_max)
            time_indices = torch.linspace(0, T_max - 1, actual_train_steps, device=wave_source.device).long()
            time_steps = time_indices.float() / T_max
        else:
            time_steps = t if t.dim() == 1 else t.view(-1)
            actual_train_steps = len(time_steps)
        
        # 对每个时间步进行训练
        for step_idx in range(actual_train_steps):
            # 当前时间步，扩展到batch维度（注意：现在batch是B*C）
            if t is None:
                t_current = time_steps[step_idx].repeat(wave_source.shape[0])
            else:
                t_current = time_steps[step_idx].repeat(wave_source.shape[0]) if time_steps.dim() == 0 else time_steps
            
            # 使用Flow Matching采样xt和ut
            t_sampled, xt, ut = SampleLocationAndConditionalFlow.run(
                self.flow_matcher,
                wave_source,
                wave_target,
                t=t_current
            )
     
            # 前向传播：输入频谱图到DiT模型
            # xt: [B*C, 2, F, T_spec] -> DiT期望输入 (N, C, F, T)
            # 记录原始时间维度大小
            original_T = xt.shape[-1]
            target_T = 690
            
            # 如果时间维度小于690，进行padding
            if original_T < target_T:
                pad_length = target_T - original_T
                xt_padded = nn.functional.pad(xt, (0, pad_length), mode='constant', value=0)
            else:
                xt_padded = xt
            
            predicted_velocity = self.model(xt_padded, t_sampled)
            
            # 裁剪predicted_velocity到原始时间维度
            if original_T < target_T:
                predicted_velocity = predicted_velocity[..., :original_T]
            
            # 计算Flow Matching损失
            flow_matching_loss = nn.functional.l1_loss(predicted_velocity, ut)
            
            # 累积损失
            total_flow_matching_loss += flow_matching_loss
        
        # 计算平均损失
        avg_flow_matching_loss = total_flow_matching_loss / actual_train_steps
        diffusion_loss = avg_flow_matching_loss * self.config.diffusion_loss_weight
        
        losses['diffusion_loss'] = diffusion_loss
        losses['flow_matching_loss'] = avg_flow_matching_loss
        losses['train_steps'] = torch.tensor(actual_train_steps, device=wave_source.device)
        
        # 梯度累积
        scaled_loss = diffusion_loss / self.config.grad_acc_step
        
        # 反向传播 - 移到循环外部
        if self.accelerator is not None:
            self.accelerator.backward(scaled_loss)
        else:
            scaled_loss.backward()
        
        # 梯度裁剪和优化器步骤
        if current_step % self.config.grad_acc_step == 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_thresh)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_thresh)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return losses
    
    def train_step_single(
        self,
        wave: torch.Tensor,
        current_step: int,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步diffusion训练（单波形输入，从噪声到波形）
        
        Args:
            wave: 目标波形 [B, T] 或 [B, C, T]
            current_step: 当前训练步数
            t: 可选的时间步，如果为None则随机采样
            
        Returns:
            包含各种损失的字典
        """
        losses = {}
        
        # 检查是否到了开始训练的步数
        if current_step < self.config.n_train_start:
            return {
                'diffusion_loss': torch.tensor(0.0, device=wave.device),
                'flow_matching_loss': torch.tensor(0.0, device=wave.device)
            }
        
        # 生成噪声作为源
        noise = torch.randn_like(wave)
        
        # 使用Flow Matching采样xt和ut
        try:
            from diffusion.utils import SampleLocationAndConditionalFlow
            t_sampled, xt, ut = SampleLocationAndConditionalFlow.run(
                self.flow_matcher,
                noise,
                wave,
                t=t
            )
        except Exception:
            try:
                from diffusion.models.utils import SampleLocationAndConditionalFlow
                t_sampled, xt, ut = SampleLocationAndConditionalFlow.run(
                    self.flow_matcher,
                    noise,
                    wave,
                    t=t
                )
            except Exception:
                # 手动实现简单版本
                if t is None:
                    t_sampled = torch.rand(wave.shape[0], device=wave.device)
                else:
                    t_sampled = t
            
            # 线性插值
            t_expanded = t_sampled.view(-1, *([1] * (wave.ndim - 1)))
            xt = (1 - t_expanded) * noise + t_expanded * wave
            ut = wave - noise
        
        # 前向传播：预测velocity
        self.optimizer.zero_grad()
        
        predicted_velocity = self.model(xt, t_sampled)
        
        # 计算Flow Matching损失
        flow_matching_loss = nn.functional.mse_loss(predicted_velocity, ut)
        diffusion_loss = flow_matching_loss * self.config.diffusion_loss_weight
        
        losses['diffusion_loss'] = diffusion_loss
        losses['flow_matching_loss'] = flow_matching_loss
        
        # 梯度累积
        scaled_loss = diffusion_loss / self.config.grad_acc_step
        
        # 反向传播
        if self.accelerator is not None:
            self.accelerator.backward(scaled_loss)
        else:
            scaled_loss.backward()
        
        # 梯度裁剪和优化器步骤
        if current_step % self.config.grad_acc_step == 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_thresh)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_thresh)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return losses
    
    @torch.no_grad()
    def inference(
        self,
        wave_source: torch.Tensor,
        num_steps: int = 10,
        method: str = 'euler',
        chunk_size_seconds: Optional[float] = None,
        num_overlap: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        推理：从源波形生成目标波形（支持分块推理）
        
        Args:
            wave_source: 源波形 [B, C, T]
            num_steps: ODE求解步数
            method: ODE求解方法 ('euler' 或 'rk4')
            chunk_size_seconds: 分块大小（秒），None则使用配置
            num_overlap: 重叠倍数，None则使用配置
            batch_size: 推理批次大小，None则使用配置
            
        Returns:
            生成的目标波形 [B, C, T]
        """
        self.model.eval()
        
        # 获取推理参数
        chunk_size_seconds = chunk_size_seconds or getattr(self.config, 'chunk_size_seconds', 8.0)
        num_overlap = num_overlap or getattr(self.config, 'num_overlap', 2)
        batch_size = batch_size or getattr(self.config, 'inference_batch_size', 1)
        sample_rate = getattr(self.config, 'sample_rate', 44100)
        
        # 保存原始信息
        B, num_channels, original_length = wave_source.shape
        device = wave_source.device
        
        # 计算chunk参数
        chunk_size = int(chunk_size_seconds * sample_rate)
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        
        # 如果使用STFT，需要先转换
        if getattr(self.config, 'use_stft', True):
            # 添加padding以处理边缘
            if original_length > 2 * border and border > 0:
                wave_source_padded = nn.functional.pad(wave_source, (border, border), mode="constant", value=0)
                # 记录padding后的长度
                padded_length = wave_source_padded.shape[-1]
            else:
                wave_source_padded = wave_source
                padded_length = original_length
            
            # 转换为频谱图 [B, C, T] -> [B*C, 2, F, T_spec]
            spec_source = self.stft(wave_source_padded)
            BC, two, F, T_spec = spec_source.shape
            
            # 初始化结果和计数器
            result = torch.zeros_like(spec_source)
            counter = torch.zeros_like(spec_source)
            
            # STFT后的窗函数需要按照hop_length缩放
            spec_chunk_size = chunk_size // self.config.hop_length + 1
            spec_fade_size = fade_size // self.config.hop_length
            spec_step = step // self.config.hop_length
            spec_windowing_array = self._get_windowing_array(spec_chunk_size, spec_fade_size, device)
            
            # 分块推理
            i = 0
            batch_data = []
            batch_locations = []
            
            while i < T_spec:
                # 提取频谱块
                part = spec_source[..., i:i + spec_chunk_size]
                chunk_len = part.shape[-1]
                
                # 统一使用常量0填充
                part = nn.functional.pad(part, (0, spec_chunk_size - chunk_len), mode="constant", value=0)
                
                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += spec_step
                
                # 处理batch
                if len(batch_data) >= batch_size or i >= T_spec:
                    arr = torch.stack(batch_data, dim=0)  # [batch, B*C, 2, F, T_spec]
                    
                    # 对batch中的每个块进行ODE求解
                    enhanced_batch = []
                    for chunk_spec in arr:
                        enhanced_spec = self._solve_ode(chunk_spec, num_steps, method)
                        enhanced_batch.append(enhanced_spec)
                    
                    enhanced_batch = torch.stack(enhanced_batch, dim=0)  # [batch, B*C, 2, F, T_spec]
                    
                    # 窗函数处理
                    window = spec_windowing_array.clone()
                    if i - spec_step == 0:  # 第一块，不需要fadein
                        window[:spec_fade_size] = 1
                    elif i >= T_spec:  # 最后一块，不需要fadeout
                        window[-spec_fade_size:] = 1
                    
                    # 累积结果
                    for j, (start, seg_len) in enumerate(batch_locations):
                        window_slice = window[..., :seg_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seg_len]
                        result[..., start:start + seg_len] += enhanced_batch[j, ..., :seg_len] * window_slice
                        counter[..., start:start + seg_len] += window_slice
                    
                    batch_data.clear()
                    batch_locations.clear()
            
            # 平均结果
            enhanced_spec = result / (counter + 1e-8)
            
            # 转换回波形，指定padding后的长度
            enhanced_wave = self.istft(enhanced_spec, length=padded_length, num_channels=num_channels)
            
            # 移除padding，精确恢复到原始长度
            if original_length > 2 * border and border > 0:
                # 确保切片后长度完全匹配原始长度
                enhanced_wave = enhanced_wave[..., border:border + original_length]
            
            # 双重保险：确保输出长度完全匹配
            if enhanced_wave.shape[-1] != original_length:
                if enhanced_wave.shape[-1] > original_length:
                    enhanced_wave = enhanced_wave[..., :original_length]
                else:
                    # 如果短了，补零（理论上不应该发生）
                    pad_len = original_length - enhanced_wave.shape[-1]
                    enhanced_wave = nn.functional.pad(enhanced_wave, (0, pad_len))
            
        else:
            # 不使用STFT的情况（直接在波形域处理）
            # 添加padding
            if original_length > 2 * border and border > 0:
                wave_source_padded = nn.functional.pad(wave_source, (border, border), mode="constant", value=0)
            else:
                wave_source_padded = wave_source
            
            # 初始化结果
            result = torch.zeros_like(wave_source_padded)
            counter = torch.zeros_like(wave_source_padded)
            
            # 创建窗函数
            windowing_array = self._get_windowing_array(chunk_size, fade_size, device)
            
            # 分块推理
            i = 0
            batch_data = []
            batch_locations = []
            
            while i < wave_source_padded.shape[-1]:
                part = wave_source_padded[..., i:i + chunk_size]
                chunk_len = part.shape[-1]
                
                # 统一使用常量0填充
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="constant", value=0)
                
                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step
                
                # 处理batch
                if len(batch_data) >= batch_size or i >= wave_source_padded.shape[-1]:
                    arr = torch.stack(batch_data, dim=0)  # [batch, B, C, T]
                    
                    # 对batch中的每个块进行ODE求解
                    enhanced_batch = []
                    for chunk_wave in arr:
                        enhanced_wave_chunk = self._solve_ode(chunk_wave, num_steps, method)
                        enhanced_batch.append(enhanced_wave_chunk)
                    
                    enhanced_batch = torch.stack(enhanced_batch, dim=0)
                    
                    # 窗函数处理
                    window = windowing_array.clone()
                    if i - step == 0:
                        window[:fade_size] = 1
                    elif i >= wave_source_padded.shape[-1]:
                        window[-fade_size:] = 1
                    
                    # 累积结果
                    for j, (start, seg_len) in enumerate(batch_locations):
                        result[..., start:start + seg_len] += enhanced_batch[j, ..., :seg_len] * window[..., :seg_len]
                        counter[..., start:start + seg_len] += window[..., :seg_len]
                    
                    batch_data.clear()
                    batch_locations.clear()
            
            # 平均结果
            enhanced_wave = result / (counter + 1e-8)
            
            # 移除padding，精确恢复到原始长度
            if original_length > 2 * border and border > 0:
                enhanced_wave = enhanced_wave[..., border:border + original_length]
            
            # 双重保险：确保输出长度完全匹配
            if enhanced_wave.shape[-1] != original_length:
                if enhanced_wave.shape[-1] > original_length:
                    enhanced_wave = enhanced_wave[..., :original_length]
                else:
                    pad_len = original_length - enhanced_wave.shape[-1]
                    enhanced_wave = nn.functional.pad(enhanced_wave, (0, pad_len))
        
        self.model.train()
        return enhanced_wave
    
    def _solve_ode(self, x: torch.Tensor, num_steps: int, method: str) -> torch.Tensor:
        """
        执行ODE求解
        
        Args:
            x: 输入 [B*C, 2, F, T] 或 [B, C, T]
            num_steps: ODE步数
            method: 求解方法
            
        Returns:
            求解结果
        """
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device)
            
            if method == 'euler':
                velocity = self.model(x, t)
                x = x + velocity * dt
            elif method == 'rk4':
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self.model(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self.model(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"未知的ODE求解方法: {method}")
        
        return x
    
    def _get_windowing_array(self, window_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
        """
        生成窗函数数组
        
        Args:
            window_size: 窗口大小
            fade_size: 淡入淡出大小
            device: 设备
            
        Returns:
            窗函数 [window_size]
        """
        fadein = torch.linspace(0, 1, fade_size, device=device)
        fadeout = torch.linspace(1, 0, fade_size, device=device)
        window = torch.ones(window_size, device=device)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window

    def save_checkpoint(self, filepath: str):
        """
        保存diffusion checkpoint
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.accelerator is not None:
            # 使用accelerate保存
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = {
                'model': unwrapped_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config': self.config.__dict__
            }
        else:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config': self.config.__dict__
            }
        
        torch.save(checkpoint, filepath)
        print(f"Diffusion checkpoint已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载diffusion checkpoint
        
        Args:
            filepath: checkpoint文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint文件不存在: {filepath}")
        
        if self.accelerator is not None:
            map_location = self.accelerator.device
        else:
            map_location = self.device or torch.device('cpu')
        
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # 加载模型状态
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Diffusion checkpoint已加载: {filepath}")
    
    def load_checkpoint_from_dict(self, checkpoint_dict: dict):
        """
        从字典加载diffusion checkpoint
        
        Args:
            checkpoint_dict: checkpoint字典
        """
        # 加载模型状态
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint_dict['model'])
        else:
            self.model.load_state_dict(checkpoint_dict['model'])
        
        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
        
        print("Diffusion checkpoint已从字典加载")
    
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> dict:
        """获取完整状态字典"""
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            model_state = unwrapped_model.state_dict()
        else:
            model_state = self.model.state_dict()
        
        return {
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
    
    def eval(self):
        """切换到评估模式"""
        self.model.eval()
    
    def train(self):
        """切换到训练模式"""
        self.model.train()
