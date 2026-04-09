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

import os
import sys
from typing import Dict, Optional, Union
import torch
import torch.nn as nn

try:
    from discriminator.models.config import DiscriminatorConfig
except ImportError:
    try:
        from .models.config import DiscriminatorConfig
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from models.config import DiscriminatorConfig


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
        self.config = DiscriminatorConfig.from_dict(config) if isinstance(config, dict) else config
            
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
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        sys.path.insert(0, current_dir)
                        from models.discriminator import Discriminator
            except ImportError:
                raise ImportError("无法导入Discriminator类，请确保models.discriminator模块存在")
            self.discriminator = Discriminator(self.config.get_mel_hparams())
        elif self.config.input_type == 'music':
            # 使用增强型MusicDiscriminator（MSD+MPD+可选SpecD，hinge损失）
            try:
                try:
                    from discriminator.models.music_discriminator import MusicDiscriminator
                except Exception:
                    from models.music_discriminator import MusicDiscriminator
            except Exception as e:
                raise ImportError(f"无法导入MusicDiscriminator：{e}")
            self.discriminator = MusicDiscriminator(**self.config.get_music_kwargs())
        else:
            # 使用HiFi-GAN风格的多尺度多周期判别器（波形域）
            try:
                try:
                    from discriminator.models.hifigan import HiFiGANDiscriminator
                except Exception:
                    from models.hifigan import HiFiGANDiscriminator
            except Exception as e:
                raise ImportError(f"无法导入HiFiGANDiscriminator：{e}")
            self.discriminator = HiFiGANDiscriminator(**self.config.get_wave_kwargs())
        
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

    def _mel_length_too_short(self, lengths: torch.Tensor) -> bool:
        if self.config.input_type != 'mel':
            return False
        return lengths.min() < self.config.min_mel_frames()
        
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
        if self._mel_length_too_short(lengths):
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
        
        if current_step < self.config.n_train_start or self._mel_length_too_short(lengths):
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


