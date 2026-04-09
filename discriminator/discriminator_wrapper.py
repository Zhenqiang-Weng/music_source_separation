"""
Discriminator wrapper with Accelerate parallel training support.
Implementation adapted from the CC-Word-Encoder discriminator design.

Usage example:
    from discriminator_wrapper import DiscriminatorWrapper

    # Initialize
    d_wrapper = DiscriminatorWrapper(
        config=d_config,
        accelerator=accelerator  # Accelerator object from accelerate
    )

    # One training step
    losses = d_wrapper.train_step(fake_data, real_data, lengths, current_step)

    # Save checkpoint
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
    Discriminator training wrapper.

    Encapsulates discriminator initialization, training, checkpoint save/load,
    and supports accelerate-based parallel training.
    """
    
    def __init__(
        self, 
        config: Union[DiscriminatorConfig, dict],
        accelerator=None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Discriminator wrapper.

        Args:
            config: Config object or config dict.
            accelerator: Accelerator object used for parallel training.
            device: Target device. Ignored when accelerator is provided.
        """
        # Parse config
        self.config = DiscriminatorConfig.from_dict(config) if isinstance(config, dict) else config
            
        self.accelerator = accelerator
        self.device = device if accelerator is None else None
        
        # Initialize components
        self._init_discriminator()
        self._init_optimizer()
        self._init_criterion()
        
        # Prepare model and optimizer when using accelerate
        if self.accelerator is not None:
            self.discriminator, self.optimizer, self.scheduler = \
                self.accelerator.prepare(self.discriminator, self.optimizer, self.scheduler)
        elif self.device is not None:
            self.discriminator = self.discriminator.to(self.device)
            
    def _init_discriminator(self):
        """Initialize discriminator model."""
        if getattr(self.config, 'input_type', 'mel') == 'mel':
            try:
                # Try several possible import paths
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
                raise ImportError("Failed to import Discriminator. Ensure models.discriminator exists.")
            self.discriminator = Discriminator(self.config.get_mel_hparams())
        elif self.config.input_type == 'music':
            # Enhanced MusicDiscriminator (MSD+MPD+optional SpecD, hinge/rank losses)
            try:
                try:
                    from discriminator.models.music_discriminator import MusicDiscriminator
                except Exception:
                    from models.music_discriminator import MusicDiscriminator
            except Exception as e:
                raise ImportError(f"Failed to import MusicDiscriminator: {e}")
            self.discriminator = MusicDiscriminator(**self.config.get_music_kwargs())
        else:
            # HiFi-GAN style multi-scale + multi-period discriminator (waveform domain)
            try:
                try:
                    from discriminator.models.hifigan import HiFiGANDiscriminator
                except Exception:
                    from models.hifigan import HiFiGANDiscriminator
            except Exception as e:
                raise ImportError(f"Failed to import HiFiGANDiscriminator: {e}")
            self.discriminator = HiFiGANDiscriminator(**self.config.get_wave_kwargs())
        
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        try:
            # Try several possible import paths
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
            raise ImportError("Failed to import NoamScheduler. Ensure models.duration exists.")
            
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
        """Initialize loss functions."""
        if self.config.input_type == 'music':
            # Music discriminator uses hinge/rank style objective
            self.criterion = None
        else:
            # mel/wave uses MSE
            self.criterion = nn.MSELoss()
        # L1 is more stable for feature matching
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
        Execute one discriminator training step.

        Args:
            fake_data: Fake data generated by the generator [B, T, Mel].
            real_data: Real data [B, T, Mel].
            lengths: Sequence lengths [B].
            current_step: Current global training step.
            return_features: Whether to return features for feature matching.

        Returns:
            Dictionary containing loss terms.
        """
        losses = {}
        
        # Check delayed discriminator start
        if current_step < self.config.n_train_start:
            return {
                'disc_loss': torch.tensor(0.0, device=fake_data.device),
                'gan_loss': torch.tensor(0.0, device=fake_data.device),
                'hdn_loss': torch.tensor(0.0, device=fake_data.device) if self.config.enable_hdn_loss else None
            }
            
        # Check sequence length constraint (mel only)
        if self._mel_length_too_short(lengths):
            return {
                'disc_loss': torch.tensor(0.0, device=fake_data.device),
                'gan_loss': torch.tensor(0.0, device=fake_data.device), 
                'hdn_loss': torch.tensor(0.0, device=fake_data.device) if self.config.enable_hdn_loss else None
            }
        
        # 1. Train discriminator to distinguish real/fake
        # self.optimizer.zero_grad()  # <-- Removed from here
        
        if self.config.input_type == 'music':
            # MusicDiscriminator: one forward call with (real, fake)
            d_out = self.discriminator(real_data, fake_data.detach())
            
            if self.config.loss_type == 'rank':
                disc_loss = self._rank_d_loss(d_out) / self.config.grad_acc_step
            else:
                disc_loss = self._hinge_d_loss(d_out) / self.config.grad_acc_step
                
            losses['disc_loss'] = disc_loss * self.config.gan_weight
        else:
            # mel/wave: separate forward passes
            if self.config.input_type == 'mel':
                d_real, _, _ = self.discriminator(real_data, lengths)
            else:
                d_real, _, _ = self.discriminator(real_data, None)
            d_real_loss = self.criterion(d_real, torch.ones_like(d_real))
            
            # Fake data pass (detach to avoid affecting generator)
            if self.config.input_type == 'mel':
                d_fake_for_disc, _, _ = self.discriminator(fake_data.detach(), lengths)
            else:
                d_fake_for_disc, _, _ = self.discriminator(fake_data.detach(), None)
            d_fake_loss = self.criterion(d_fake_for_disc, torch.zeros_like(d_fake_for_disc))
            
            # Total discriminator loss
            disc_loss = (d_real_loss + d_fake_loss) / self.config.grad_acc_step
            losses['disc_loss'] = disc_loss * self.config.gan_weight
        
        # Backward pass for discriminator
        if self.accelerator is not None:
            self.accelerator.backward(disc_loss)
        else:
            disc_loss.backward()
            
        # Gradient clipping and optimizer step
        if current_step % self.config.grad_acc_step == 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_thresh)
            else:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_thresh)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()  # Moved here
            
        # 2. Compute generator adversarial loss (no backward here)
        with torch.set_grad_enabled(True):  # Gradients are needed for generator training
            if self.config.input_type == 'music':
                # MusicDiscriminator
                d_out = self.discriminator(real_data, fake_data)
                
                if self.config.loss_type == 'rank':
                    gan_loss = self._rank_g_loss(d_out)
                else:
                    gan_loss = self._hinge_g_loss(d_out)
                    
                losses['gan_loss'] = gan_loss * self.config.gan_weight
                h_fake = d_out  # cache for FM
            else:
                if self.config.input_type == 'mel':
                    d_fake_for_gen, start_frames_wins, h_fake = self.discriminator(fake_data, lengths)
                else:
                    d_fake_for_gen, start_frames_wins, h_fake = self.discriminator(fake_data, None)
                gan_loss = self.criterion(d_fake_for_gen, torch.ones_like(d_fake_for_gen))
                losses['gan_loss'] = gan_loss * self.config.gan_weight
            
        # 3. Feature matching loss
        if self.config.enable_hdn_loss and return_features:
            with torch.no_grad():
                if self.config.input_type == 'music':
                    # FM loss for MusicDiscriminator
                    hdn_loss = self._feature_matching_loss(h_fake)
                    losses['hdn_loss'] = hdn_loss * self.config.fml_weight
                else:
                    if self.config.input_type == 'mel':
                        _, _, h_real = self.discriminator(real_data, lengths, start_frames_wins=start_frames_wins)
                    else:
                        _, _, h_real = self.discriminator(real_data, None)
                    
                    # Flatten feature hierarchy
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
                            # Fallback: average pairwise FM loss if stacking fails
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
        Get adversarial and feature-matching losses for generator training.

        Args:
            fake_data: Generated fake data [B, T, Mel] or [B, T] (wave/music).
            real_data: Real data [B, T, Mel] or [B, T].
            lengths: Sequence lengths [B] (required for mel, ignored otherwise).
            current_step: Current global training step.

        Returns:
            Dictionary containing generator-side losses.
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
            # mel/wave: standard flow
            if self.config.input_type == 'mel':
                d_fake, start_frames_wins, h_fake = self.discriminator(fake_data, lengths)
            else:
                d_fake, start_frames_wins, h_fake = self.discriminator(fake_data, None)
            gan_loss = self.criterion(d_fake, torch.ones_like(d_fake))
            losses['gan_loss'] = gan_loss * self.config.gan_weight
            
            # Feature matching loss
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
        Save discriminator checkpoint.

        Args:
            filepath: Destination path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.accelerator is not None:
            # Save using accelerate-unwrapped model
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
        print(f"Discriminator checkpoint saved to: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        Load discriminator checkpoint.

        Args:
            filepath: Checkpoint file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
        if self.accelerator is not None:
            map_location = self.accelerator.device
        else:
            map_location = self.device or torch.device('cpu')
            
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            unwrapped_model.load_state_dict(checkpoint['model'])
        else:
            self.discriminator.load_state_dict(checkpoint['model'])
            
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Discriminator checkpoint loaded: {filepath}")
        
    def load_checkpoint_from_dict(self, checkpoint_dict: dict):
        """
        Load discriminator checkpoint from a dictionary.

        Args:
            checkpoint_dict: Checkpoint dictionary.
        """
        # Load model state
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.discriminator)
            unwrapped_model.load_state_dict(checkpoint_dict['model'])
        else:
            self.discriminator.load_state_dict(checkpoint_dict['model'])
            
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
        
        print("Discriminator checkpoint loaded from dictionary")
        
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
        
    def state_dict(self) -> dict:
        """Get full state dictionary."""
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
        """Switch to evaluation mode."""
        self.discriminator.eval()
        
    def train(self): 
        """Switch to training mode."""
        self.discriminator.train()


