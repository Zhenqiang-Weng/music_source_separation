"""
Discriminator包装类 - 支持Accelerate并行训练
基于CC-Word-Encoder项目的discriminator实现

使用示例:
    from discriminator_wrapper import DiscriminatorWrapper
    
    # 初始化
    d_wrapper = DiscriminatorWrapper(
        config=d_config,
        accelerator=accelerator  # 从accelerate导入的accelerator对象
    )
    
    # 训练步骤
    losses = d_wrapper.train_step(fake_data, real_data, lengths, current_step)
    
    # 保存checkpoint
    d_wrapper.save_checkpoint("path/to/checkpoint.pth")
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Optional, Union
from dataclasses import dataclass
import sys



@dataclass
class DiscriminatorConfig:
    """Discriminator配置类"""
    # 模型参数
    time_lengths: list = None  # 时间窗口长度，如[64,64,64]
    freq_lengths: list = None  # 频带长度，如[40,40,40] 
    kernel_size: int = 3
    hidden_size: int = 128
    n_mel_channels: int = 128  # mel频谱通道数
    # 输入类型：
    # 'mel' 使用本项目的频带多窗口判别器
    # 'wave' 使用HiFi-GAN风格MSD+MPD（简洁版）
    # 'music' 使用增强型MusicDiscriminator（MSD+MPD+可选SpecD，支持hinge损失）
    input_type: str = 'mel'
    # HiFi-GAN/Music 参数（当 input_type 为 'wave' 或 'music' 时有效）
    mpd_periods: list = None  # 例如 [2,3,5,7,11,13]
    msd_pool_scales: list = None  # 例如 [1,2,4,8]（代替 ms_poolings）
    ms_poolings: list = None  # 兼容字段，若设置则覆盖 msd_pool_scales
    # Music判别器专用
    enable_music_spec: bool = False  # 是否启用多分辨率频谱判别分支
    mr_stft_cfg: list = None  # 多分辨率STFT配置，例如 [dict(n_fft=1024, hop=256, win=1024), ...]
    # 损失函数类型：'mse'（LSGAN，默认）或 'hinge'（仅 input_type='music' 时）或 'rank'（偏好对loss）
    loss_type: str = 'rank'
    rank_margin: float = 0.0  # Rank loss的边距参数，仅当 loss_type='rank' 时有效
    
    # 训练参数
    n_train_start: int = 0  # 延迟启动步数
    # 新：分别控制对抗损失与特征匹配损失的权重
    gan_weight: float = 0.5  # 对抗损失权重（原 gan_loss_lambda）
    fml_weight: float = 1.0 # 特征匹配损失权重（Feature Matching Loss）
    # 兼容：若仍传入 gan_loss_lambda，将同时覆盖 gan_weight 与 fml_weight
    gan_loss_lambda: Optional[float] = None  # 兼容字段，不再直接使用
    enable_hdn_loss: bool = True  # 是否启用特征匹配损失
    
    # 优化器参数
    lr: float = 0.0001
    betas: list = None  # [0.9, 0.98]
    eps: float = 1e-9
    weight_decay: float = 0.0
    
    # 调度器参数  
    n_warmup: int = 2000
    init_scale: float = 0.25
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1
    
    def __post_init__(self):
        if self.time_lengths is None:
            self.time_lengths = [64, 64, 64]
        if self.freq_lengths is None:
            self.freq_lengths = [64, 64, 64]
        if self.betas is None:
            self.betas = [0.9, 0.98]
        if self.mpd_periods is None:
            self.mpd_periods = [2, 3, 5, 7, 11, 13] if self.input_type == 'music' else [2, 3, 5, 7, 11]
            # self.mpd_periods = [2, 3, 5, 7, 11] if self.input_type == 'music' else [2, 3, 5, 7, 11]
        # msd_pool_scales 和 ms_poolings 兼容处理
        if self.ms_poolings is not None and self.msd_pool_scales is None:
            self.msd_pool_scales = self.ms_poolings
        if self.msd_pool_scales is None:
            self.msd_pool_scales = [1, 2, 4, 8] if self.input_type == 'music' else [1, 2, 4]
            # self.msd_pool_scales = [1, 2, 4] if self.input_type == 'music' else [1, 2, 4]
        # Music判别器的多分辨率STFT配置
        if self.mr_stft_cfg is None:
            self.mr_stft_cfg = [
                dict(n_fft=1024, hop=256, win=1024),
                dict(n_fft=2048, hop=512, win=2048),
                dict(n_fft=4096, hop=1024, win=4096),
            ]
        # 兼容旧参数：如果提供了 gan_loss_lambda，则同时设置 gan_weight 和 fml_weight
        if self.gan_loss_lambda is not None:
            try:
                val = float(self.gan_loss_lambda)
            except Exception:
                val = 0.1
            self.gan_weight = val
            self.fml_weight = val


class DiscriminatorWrapper:
    """
    Discriminator训练包装类
    
    封装了discriminator的初始化、训练、保存等功能，
    并支持accelerate并行训练
    """
    
    def __init__(
        self, 
        config: Union[DiscriminatorConfig, dict],
        accelerator=None,
        device: Optional[torch.device] = None
    ):
        """
        初始化Discriminator包装类
        
        Args:
            config: 配置对象或字典
            accelerator: accelerate的Accelerator对象，用于并行训练
            device: 设备，如果提供accelerator则忽略此参数
        """
        # 配置处理
        if isinstance(config, dict):
            self.config = DiscriminatorConfig(**config)
        else:
            self.config = config
            
        self.accelerator = accelerator
        self.device = device if accelerator is None else None
        
        # 初始化组件
        self._init_discriminator()
        self._init_optimizer()
        self._init_criterion()
        
        # 如果使用accelerate，准备模型和优化器
        if self.accelerator is not None:
            self.discriminator, self.optimizer, self.scheduler = \
                self.accelerator.prepare(self.discriminator, self.optimizer, self.scheduler)
        elif self.device is not None:
            self.discriminator = self.discriminator.to(self.device)
            
    def _init_discriminator(self):
        """初始化discriminator模型"""
        if getattr(self.config, 'input_type', 'mel') == 'mel':
            try:
                # 尝试多个可能的导入路径
                try:
                    from discriminator.models.discriminator import Discriminator
                except ImportError:
                    try:
                        from .models.discriminator import Discriminator
                    except ImportError:
                        import sys
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        sys.path.insert(0, current_dir)
                        from models.discriminator import Discriminator
            except ImportError:
                raise ImportError("无法导入Discriminator类，请确保models.discriminator模块存在")
            self.discriminator = Discriminator(self.config)
        elif self.config.input_type == 'music':
            # 使用增强型MusicDiscriminator（MSD+MPD+可选SpecD，hinge损失）
            try:
                try:
                    from discriminator.models.music_discriminator import MusicDiscriminator
                except Exception:
                    from models.music_discriminator import MusicDiscriminator
            except Exception as e:
                raise ImportError(f"无法导入MusicDiscriminator：{e}")
            self.discriminator = MusicDiscriminator(
                sample_rate=44100,
                mpd_periods=tuple(self.config.mpd_periods),
                msd_pool_scales=tuple(self.config.msd_pool_scales),
                enable_spec=self.config.enable_music_spec,
                mr_stft_cfg=self.config.mr_stft_cfg,
            )
        else:
            # 使用HiFi-GAN风格的多尺度多周期判别器（波形域）
            try:
                try:
                    from discriminator.models.hifigan import HiFiGANDiscriminator
                except Exception:
                    from models.hifigan import HiFiGANDiscriminator
            except Exception as e:
                raise ImportError(f"无法导入HiFiGANDiscriminator：{e}")
            self.discriminator = HiFiGANDiscriminator(
                periods=tuple(self.config.mpd_periods),
                poolings=tuple(self.config.msd_pool_scales),
            )
        
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        try:
            # 尝试多个可能的导入路径
            try:
                from discriminator.models.duration import NoamScheduler
            except ImportError:
                try:
                    from models.duration import NoamScheduler
                except ImportError:
                    import sys
                    import os
                    # 添加cc-word-encoder目录到路径
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.insert(0, current_dir)
                    from models.duration import NoamScheduler
        except ImportError:
            raise ImportError("无法导入NoamScheduler类，请确保models.duration模块存在")
            
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = NoamScheduler(
            self.optimizer,
            n_warmup=self.config.n_warmup,
            init_scale=self.config.init_scale
        )
        
    def _init_criterion(self):
        """初始化损失函数"""
        if self.config.input_type == 'music':
            # Music判别器使用hinge损失
            self.criterion = None  # 用 hinge 损失函数替代
        else:
            # mel/wave 使用 MSE
            self.criterion = nn.MSELoss()
        # 特征匹配使用L1更稳
        self.fm_criterion = nn.L1Loss()
        
    def _hinge_d_loss(self, d_out):
        """Hinge discriminator loss for MusicDiscriminator."""
        loss = 0.0
        count = 0
        for item in d_out:
            real_logit = item['real_logit']
            fake_logit = item['fake_logit']
            loss += torch.relu(1.0 - real_logit).mean() + torch.relu(1.0 + fake_logit).mean()
            count += 1
        return loss / max(count, 1)
    
    def _hinge_g_loss(self, d_out):
        """Hinge generator loss for MusicDiscriminator."""
        loss = 0.0
        count = 0
        for item in d_out:
            fake_logit = item['fake_logit']
            loss += torch.relu(1.0 - fake_logit).mean()
            count += 1
        return loss / max(count, 1)
    
    def _rank_d_loss(self, d_out):
        """Rank discriminator loss: -log(sigmoid(real - fake))."""
        loss = 0.0
        count = 0
        for item in d_out:
            real_logit = item['real_logit']
            fake_logit = item['fake_logit']
            # softplus(fake - real)
            loss += torch.nn.functional.softplus(fake_logit - real_logit + self.config.rank_margin).mean()
            count += 1
        return loss / max(count, 1)

    def _rank_g_loss(self, d_out):
        """Rank generator loss: -log(sigmoid(fake - real))."""
        loss = 0.0
        count = 0
        for item in d_out:
            real_logit = item['real_logit']
            fake_logit = item['fake_logit']
            # softplus(real - fake)
            loss += torch.nn.functional.softplus(real_logit - fake_logit + self.config.rank_margin).mean()
            count += 1
        return loss / max(count, 1)
    
    def _feature_matching_loss(self, d_out):
        """L1 feature matching loss."""
        loss = 0.0
        count = 0
        for item in d_out:
            real_fmaps = item['real_fmaps']
            fake_fmaps = item['fake_fmaps']
            for rf, ff in zip(real_fmaps[:-1], fake_fmaps[:-1]):
                loss += self.fm_criterion(ff, rf.detach())
                count += 1
        return loss / max(count, 1)
        
    def train_step(
        self, 
        fake_data: torch.Tensor,
        real_data: torch.Tensor, 
        lengths: torch.Tensor,
        current_step: int,
        return_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步discriminator训练
        
        Args:
            fake_data: 生成器生成的假数据 [B, T, Mel]
            real_data: 真实数据 [B, T, Mel]  
            lengths: 序列长度 [B]
            current_step: 当前训练步数
            return_features: 是否返回特征用于特征匹配损失
            
        Returns:
            包含各种损失的字典
        """
        losses = {}
        
        # 检查是否到了开始训练discriminator的步数
        if current_step < self.config.n_train_start:
            return {
                'disc_loss': torch.tensor(0.0, device=fake_data.device),
                'gan_loss': torch.tensor(0.0, device=fake_data.device),
                'hdn_loss': torch.tensor(0.0, device=fake_data.device) if self.config.enable_hdn_loss else None
            }
            
        # 检查序列长度是否满足要求（mel判别器需要时间窗口；wave模式忽略长度约束）
        if self.config.input_type == 'mel' and lengths.min() < max(self.config.time_lengths):
            return {
                'disc_loss': torch.tensor(0.0, device=fake_data.device),
                'gan_loss': torch.tensor(0.0, device=fake_data.device), 
                'hdn_loss': torch.tensor(0.0, device=fake_data.device) if self.config.enable_hdn_loss else None
            }
        
        # 1. 训练Discriminator - 区分真假数据
        # self.optimizer.zero_grad()  # <-- Removed from here
        
        if self.config.input_type == 'music':
            # MusicDiscriminator: 一次前向调用 (real, fake)，返回结构化输出
            d_out = self.discriminator(real_data, fake_data.detach())
            
            if self.config.loss_type == 'rank':
                disc_loss = self._rank_d_loss(d_out) / self.config.grad_acc_step
            else:
                disc_loss = self._hinge_d_loss(d_out) / self.config.grad_acc_step
                
            losses['disc_loss'] = disc_loss * self.config.gan_weight
        else:
            # mel/wave: 分别前向
            if self.config.input_type == 'mel':
                d_real, _, _ = self.discriminator(real_data, lengths)
            else:
                d_real, _, _ = self.discriminator(real_data, None)
            d_real_loss = self.criterion(d_real, torch.ones_like(d_real))
            
            # 假数据判别（detach以避免影响生成器）
            if self.config.input_type == 'mel':
                d_fake_for_disc, _, _ = self.discriminator(fake_data.detach(), lengths)
            else:
                d_fake_for_disc, _, _ = self.discriminator(fake_data.detach(), None)
            d_fake_loss = self.criterion(d_fake_for_disc, torch.zeros_like(d_fake_for_disc))
            
            # Discriminator总损失
            disc_loss = (d_real_loss + d_fake_loss) / self.config.grad_acc_step
            losses['disc_loss'] = disc_loss * self.config.gan_weight
        
        # 反向传播discriminator
        if self.accelerator is not None:
            self.accelerator.backward(disc_loss)
        else:
            disc_loss.backward()
            
        # 梯度裁剪和优化器步骤
        if current_step % self.config.grad_acc_step == 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_thresh)
            else:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_thresh)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()  # Moved here
            
        # 2. 计算生成器的对抗损失（不反向传播）
        with torch.set_grad_enabled(True):  # 需要梯度用于生成器训练
            if self.config.input_type == 'music':
                # MusicDiscriminator
                d_out = self.discriminator(real_data, fake_data)
                
                if self.config.loss_type == 'rank':
                    gan_loss = self._rank_g_loss(d_out)
                else:
                    gan_loss = self._hinge_g_loss(d_out)
                    
                losses['gan_loss'] = gan_loss * self.config.gan_weight
                h_fake = d_out  # 保存用于FM
            else:
                if self.config.input_type == 'mel':
                    d_fake_for_gen, start_frames_wins, h_fake = self.discriminator(fake_data, lengths)
                else:
                    d_fake_for_gen, start_frames_wins, h_fake = self.discriminator(fake_data, None)
                gan_loss = self.criterion(d_fake_for_gen, torch.ones_like(d_fake_for_gen))
                losses['gan_loss'] = gan_loss * self.config.gan_weight
            
        # 3. 特征匹配损失
        if self.config.enable_hdn_loss and return_features:
            with torch.no_grad():
                if self.config.input_type == 'music':
                    # MusicDiscriminator 的 FM 损失
                    hdn_loss = self._feature_matching_loss(h_fake)
                    losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                else:
                    if self.config.input_type == 'mel':
                        _, _, h_real = self.discriminator(real_data, lengths, start_frames_wins=start_frames_wins)
                    else:
                        _, _, h_real = self.discriminator(real_data, None)
                    
                    # 展平特征层次结构
                    h_fake_flat = [h_in for h_out in h_fake for h_in in h_out]
                    h_real_flat = [h_in for h_out in h_real for h_in in h_out]
                    
                    if h_fake_flat and h_real_flat:
                        try:
                            hdn_loss = self.fm_criterion(
                                torch.stack(h_fake_flat, dim=-1), 
                                torch.stack(h_real_flat, dim=-1)
                            )
                            losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                        except RuntimeError:
                            # 如果stack失败，使用平均特征匹配损失
                            hdn_loss = sum(
                                self.fm_criterion(hf, hr) 
                                for hf, hr in zip(h_fake_flat, h_real_flat)
                                if hf.shape == hr.shape
                            ) / len(h_fake_flat)
                            losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                    else:
                        losses['hdn_loss'] = torch.tensor(0.0, device=fake_data.device)
        else:
            losses['hdn_loss'] = None if not self.config.enable_hdn_loss else torch.tensor(0.0, device=fake_data.device)

        return losses
        
    def get_generator_losses(
        self,
        fake_data: torch.Tensor,
        real_data: torch.Tensor,
        lengths: torch.Tensor,
        current_step: int
    ) -> Dict[str, torch.Tensor]:
        """
        获取生成器的对抗损失和特征匹配损失（用于生成器训练）
        
        Args:
            fake_data: 生成器生成的假数据 [B, T, Mel] 或 [B, T] (wave/music)
            real_data: 真实数据 [B, T, Mel] 或 [B, T]
            lengths: 序列长度 [B]（mel模式需要；wave/music模式忽略）
            current_step: 当前训练步数
            
        Returns:
            包含生成器损失的字典
        """
        losses = {}
        
        if current_step < self.config.n_train_start or (self.config.input_type == 'mel' and lengths.min() < max(self.config.time_lengths)):
            return {
                'gan_loss': torch.tensor(0.0, device=fake_data.device),
                'hdn_loss': torch.tensor(0.0, device=fake_data.device) if self.config.enable_hdn_loss else None
            }
        
        if self.config.input_type == 'music':
            # MusicDiscriminator
            d_out = self.discriminator(real_data, fake_data)
            
            if self.config.loss_type == 'rank':
                gan_loss = self._rank_g_loss(d_out)
            else:
                gan_loss = self._hinge_g_loss(d_out)
                
            losses['gan_loss'] = gan_loss * self.config.gan_weight
            
            if self.config.enable_hdn_loss:
                hdn_loss = self._feature_matching_loss(d_out)
                losses['hdn_loss'] = hdn_loss * self.config.fml_weight
            else:
                losses['hdn_loss'] = None
        else:
            # mel/wave: 原有流程
            if self.config.input_type == 'mel':
                d_fake, start_frames_wins, h_fake = self.discriminator(fake_data, lengths)
            else:
                d_fake, start_frames_wins, h_fake = self.discriminator(fake_data, None)
            gan_loss = self.criterion(d_fake, torch.ones_like(d_fake))
            losses['gan_loss'] = gan_loss * self.config.gan_weight
            
            # 特征匹配损失
            if self.config.enable_hdn_loss:
                with torch.no_grad():
                    if self.config.input_type == 'mel':
                        _, _, h_real = self.discriminator(real_data, lengths, start_frames_wins=start_frames_wins)
                    else:
                        _, _, h_real = self.discriminator(real_data, None)
                    
                h_fake_flat = [h_in for h_out in h_fake for h_in in h_out]  
                h_real_flat = [h_in for h_out in h_real for h_in in h_out]
                
                if h_fake_flat and h_real_flat:
                    try:
                        hdn_loss = self.fm_criterion(
                            torch.stack(h_fake_flat, dim=-1),
                            torch.stack(h_real_flat, dim=-1)
                        )
                        losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                    except RuntimeError:
                        hdn_loss = sum(
                            self.fm_criterion(hf, hr)
                            for hf, hr in zip(h_fake_flat, h_real_flat) 
                            if hf.shape == hr.shape
                        ) / len(h_fake_flat)
                        losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                else:
                    losses['hdn_loss'] = torch.tensor(0.0, device=fake_data.device)
            else:
                losses['hdn_loss'] = None
            
        return losses
        
    def save_checkpoint(self, filepath: str):
        """
        保存discriminator checkpoint
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.accelerator is not None:
            # 使用accelerate保存
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            checkpoint = {
                'model': unwrapped_model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 
                'scheduler': self.scheduler.state_dict(),
                'config': self.config.__dict__
            }
        else:
            checkpoint = {
                'model': self.discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(), 
                'config': self.config.__dict__
            }
            
        torch.save(checkpoint, filepath)
        print(f"Discriminator checkpoint已保存到: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        加载discriminator checkpoint
        
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
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            unwrapped_model.load_state_dict(checkpoint['model'])
        else:
            self.discriminator.load_state_dict(checkpoint['model'])
            
        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Discriminator checkpoint已加载: {filepath}")
        
    def load_checkpoint_from_dict(self, checkpoint_dict: dict):
        """
        从字典加载discriminator checkpoint
        
        Args:
            checkpoint_dict: checkpoint字典
        """
        # 加载模型状态
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            unwrapped_model.load_state_dict(checkpoint_dict['model'])
        else:
            self.discriminator.load_state_dict(checkpoint_dict['model'])
            
        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
        
        print("Discriminator checkpoint已从字典加载")
        
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
        
    def state_dict(self) -> dict:
        """获取完整状态字典"""
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            model_state = unwrapped_model.state_dict()
        else:
            model_state = self.discriminator.state_dict()
            
        return {
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        
    def eval(self):
        """切换到评估模式"""
        self.discriminator.eval()
        
    def train(self): 
        """切换到训练模式"""
        self.discriminator.train()


