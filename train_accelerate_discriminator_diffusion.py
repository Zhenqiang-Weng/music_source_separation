# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

# Read more here:
# https://huggingface.co/docs/accelerate/index
# 改进: 修复多卡训练问题

import argparse
import soundfile as sf
import numpy as np
import time
import glob
from tqdm.auto import tqdm
import os
import torch
import wandb
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from utils.dataset import MSSDataset
from utils.model_utils import demix, prefer_target_instrument, load_not_compatible_weights
from utils.metrics import sdr, get_metrics
from utils.settings import manual_seed, get_model_from_config
from utils.losses import masked_loss

import torchaudio
from discriminator.discriminator_wrapper import DiscriminatorWrapper, DiscriminatorConfig
from diffusion.diffusion_wrapper import DiffusionWrapper, DiffusionConfig

import warnings

warnings.filterwarnings("ignore")

metric_list = ["sdr", "si_sdr", "l1_freq", "bleedless", "fullness"]

def valid(model, valid_loader, args, config, device, verbose=False, diffusion_wrapper=None, current_step=None):
    instruments = prefer_target_instrument(config)

    # 改为字典的字典结构: {metric: {instr: []}}
    all_metrics = {}
    for metric in metric_list:
        all_metrics[metric] = {}
        for instr in instruments:
            all_metrics[metric][instr] = []

    all_mixtures_path = valid_loader
    if verbose:
        all_mixtures_path = tqdm(valid_loader, ncols=200)

    pbar_dict = {}
    for path_list in all_mixtures_path:
        path = path_list[0]
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        res = demix(config, model, mix.T, device, model_type=args.model_type)  # 返回 dict: {instr: [C, T]}

        for stem_idx, instr in enumerate(instruments):
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
            else:
                # other is actually instrumental
                track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                track = mix - track
            
            # 如果使用扩散模型进行后处理/增强
            if diffusion_wrapper is not None and (diffusion_wrapper[stem_idx] is not None and current_step >= diffusion_wrapper[stem_idx].config.n_train_start):
                diff_model = diffusion_wrapper[stem_idx]
                if diff_model is not None:
                    # res[instr]: [C, T] numpy array
                    # 转换为 torch tensor: [C, T] -> [1, C, T]
                    source_wave = torch.from_numpy(res[instr]).unsqueeze(0).to(device)
                    
                    # 使用扩散模型推理
                    with torch.no_grad():
                        enhanced_wave = diff_model.inference(
                            source_wave,  # [1, C, T]
                            num_steps=10,  # 固定为10步
                            method='euler'
                        )
                    
                    # 转换回numpy: [1, C, T] -> [C, T]
                    res[instr] = enhanced_wave.squeeze(0).cpu().numpy()
            
            # 用于计算指标: res[instr].T 将 [C, T] 转为 [T, C]
            references = np.expand_dims(track, axis=0)  # [T, C] -> [1, T, C]
            estimates = np.expand_dims(res[instr].T, axis=0)  # [C, T] -> [T, C] -> [1, T, C]

            results = get_metrics(metric_list, references, estimates, mix)
            # results 是一个列表: [(metric_name, value), ...]
            for metric_name, value in results:
                single_val = torch.tensor([value], device=device, dtype=torch.float32)
                all_metrics[metric_name][instr].append(single_val)
                pbar_dict['{}_{}'.format(metric_name, instr)] = value
        if verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    return all_metrics


class MSSValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        all_mixtures_path = []
        for valid_path in args.valid_path:
            part = sorted(glob.glob(valid_path + '/*/mixture.wav'))
            if len(part) == 0:
                print('No validation data found in: {}'.format(valid_path))
            all_mixtures_path += part

        self.list_of_files = all_mixtures_path

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        return self.list_of_files[index]


def train_model(args):
    # unused kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str, help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str, help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0,1,2,3], help='list of gpu ids')
    parser.add_argument("--use_multistft_loss", action='store_true', help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    # wandb params
    parser.add_argument("--wandb_project", type=str, default='msst-accelerate', help='wandb project name')
    parser.add_argument("--wandb_name", type=str, default='', help='wandb run name')
    # discriminator training params
    parser.add_argument("--use_discriminator", action='store_true', help="Use discriminator for training")
    parser.add_argument("--discriminator_start_check_point", type=str, default='', help="Initial checkpoint to start discriminator training")
    # frequency weighted loss
    parser.add_argument("--use_frequency_weighted_loss", action='store_true', help="Use frequency weighted loss along with other losses")
    # load first mask estimator to others
    parser.add_argument("--copy_first_mask_estimator", action='store_true', help="Copy first mask estimator weights to other estimators after loading checkpoint")
    # frozen estimators
    parser.add_argument("--unfreeze_mask_estimators", action='store_true', help="Unfreeze mask estimators except the first one")
    parser.add_argument("--estimator_unfreeze_indexes", nargs='+', type=int, default=[0,1], help="List of mask estimator indexes to unfreeze (0-based)")
    # diffusion params
    parser.add_argument("--use_diffusion", action='store_true', help="Use diffusion model for training")
    parser.add_argument("--diffusion_model_path", type=str, default='', help="Path to diffusion model checkpoint")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    manual_seed(args.seed + int(time.time()))
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # Fix possible slow down with dilation convolutions
    
    # 修复1: 添加异常处理，避免多次设置spawn导致错误
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        accelerator.print(f"Warning: multiprocessing start method already set: {e}")

    model, config = get_model_from_config(args.model_type, args.config_path)
    accelerator.print("Instruments: {}".format(config.training.instruments))
    
    # 修复2: 添加调试信息，显示分布式训练信息
    accelerator.print(f"[DEBUG] Number of processes: {accelerator.num_processes}")
    accelerator.print(f"[DEBUG] Device: {accelerator.device}")
    accelerator.print(f"[DEBUG] Is main process: {accelerator.is_main_process}")

    os.makedirs(args.results_path, exist_ok=True)

    device_ids = args.device_ids
    batch_size = config.training.batch_size

    # wandb
    if accelerator.is_main_process:
        if args.wandb_key is not None and args.wandb_key.strip() != '':
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project, name=args.wandb_name, config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size})
        else:
            wandb.init(project=args.wandb_project, name=args.wandb_name, mode='offline', config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size})
    else:
        wandb.init(mode='disabled')

    # Fix for num of steps
    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, 'metadata_{}.pkl'.format(args.dataset_type)),
        dataset_type=args.dataset_type,
        verbose=accelerator.is_main_process,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
    )

    # 修复3: 检查点加载应该在prepare之前
    if args.start_check_point != '':
        accelerator.print('Start from checkpoint: {}'.format(args.start_check_point))
        if 0:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            # 在prepare前加载，确保权重正确应用
            checkpoint = torch.load(args.start_check_point, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            if args.copy_first_mask_estimator:
                if hasattr(model, 'copy_first_mask_estimator_to_others'):
                    model.copy_first_mask_estimator_to_others()
            if args.unfreeze_mask_estimators:
                if hasattr(model, 'frozen_all') and hasattr(model, 'unfreeze_mask_estimators_by_indexes'):
                    model.frozen_all()
                    model.unfreeze_mask_estimators_by_indexes(args.estimator_unfreeze_indexes)
            accelerator.print(f"Checkpoint loaded successfully")
            
    # Discriminator setup - 从配置文件读取
    discriminators = []
    gan_models = []
    if args.use_discriminator:
        # 从配置文件读取 gan_model 列表
        gan_models_raw = getattr(config.training, 'gan_model', None)
        num_stems = len(config.training.instruments)
        
        if gan_models_raw is None:
            # 默认使用 mel 判别器
            accelerator.print("Warning: gan_model not found in config, using default 'mel' for all stems")
            gan_models = ['mel'] * num_stems
        elif len(gan_models_raw) != num_stems:
            raise ValueError(f"gan_model length ({len(gan_models_raw)}) must match number of instruments ({num_stems})")
        else:
            gan_models = gan_models_raw
        
        accelerator.print(f"Creating discriminators for {num_stems} stems with types: {gan_models}")
        
        for idx, (instr, input_type) in enumerate(zip(config.training.instruments, gan_models)):
            if input_type is None or input_type.lower() == 'none':
                # 不使用判别器
                accelerator.print(f"Stem '{instr}': No discriminator (None)")
                discriminators.append(None)
                continue
            
            # 根据 input_type 创建配置
            disc_cfg = DiscriminatorConfig(
                input_type=input_type,
            )
            
            disc_wrapper = DiscriminatorWrapper(disc_cfg, accelerator)
            
            # 加载检查点（如果提供）
            if args.discriminator_start_check_point != '':
                ckpt_path = os.path.join(args.discriminator_start_check_point, f'{instr}.pth')
                if os.path.exists(ckpt_path):
                    disc_wrapper.load_checkpoint(ckpt_path)
                    accelerator.print(f"Loaded discriminator checkpoint for {instr} from {ckpt_path}")
                else:
                    accelerator.print(f"Warning: checkpoint not found for {instr} at {ckpt_path}")
            
            disc_wrapper.train()
            accelerator.print(f"Stem '{instr}': Discriminator type '{input_type}'")
            discriminators.append(disc_wrapper)
        
        # Prepare all discriminators (skip None)
        discriminators = [accelerator.prepare(d) if d is not None else None for d in discriminators]

    # MelSpectrogram for discriminator (仅当使用 mel 类型时需要)
    mel = None
    to_db = None
    if args.use_discriminator and any(m == 'mel' for m in gan_models if m is not None):
        sr = 44100
        n_fft = 2048
        hop = 441
        n_mels = 128

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0
        ).to(accelerator.device)

        to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=130).to(accelerator.device)

    def wave_BC_to_mel(w):  # w: [B, C, T]
        # 梅尔：输出 [B, C, n_mels, F]
        m = mel(w) + 1e-5
        m = to_db(m)                     # [B, C, n_mels, F]
        m = m.permute(0, 1, 3, 2)        # [B, C, F, n_mels]
        B, C, F, M = m.shape
        return m.reshape(B * C, F, M)    # [B*C, F, n_mels] = [B*C, frames, n_mels]

    def samp_to_frames(Ls, C):
        Lf = torch.ceil(Ls.float() / hop).long()
        return Lf.repeat_interleave(C)
    
    def wave_BC_to_wave_B(w):  # w: [B, C, T]
        # 将双声道音频转换为单个batch维度: [B*C, T]
        B, C, T = w.shape
        return w.reshape(B * C, T)
    # End of MelSpectrogram for discriminator
    
    
    # Diffusion model setup
    diffusion_wrappers = []
    if args.use_diffusion:
        diffusion_model_raw = getattr(config.training, 'diffusion_model', None)
        num_stems = len(config.training.instruments)
        diffusion_models = []
        if diffusion_model_raw is None:
            accelerator.print("Warning: diffusion_model not found in config, using default 'dit' for all stems")
            diffusion_models = ['dit'] * num_stems
        elif len(diffusion_model_raw) != num_stems:
            raise ValueError(f"diffusion_model length ({len(diffusion_model_raw)}) must match number of instruments ({num_stems})")
        else:
            diffusion_models = diffusion_model_raw
        
        accelerator.print(f"Creating diffusion models for {num_stems} stems with types: {diffusion_models}")
        
        for idx, (instr, diffusion_type) in enumerate(zip(config.training.instruments, diffusion_models)):
            if diffusion_type is None or diffusion_type.lower() == 'none':
                diffusion_wrappers.append(None)
                accelerator.print(f"Stem '{instr}': No diffusion model (None)")
                continue
            
            diff_cfg = DiffusionConfig()
            
            diff_wrapper = DiffusionWrapper(diff_cfg, accelerator)
            
            # 加载检查点（如果提供）
            if args.diffusion_model_path != '':
                ckpt_path = os.path.join(args.diffusion_model_path, f'{instr}_diffusion.pth')
                if os.path.exists(ckpt_path):
                    diff_wrapper.load_checkpoint(ckpt_path)
                    accelerator.print(f"Loaded diffusion checkpoint for {instr} from {ckpt_path}")
                else:
                    accelerator.print(f"Warning: diffusion checkpoint not found for {instr} at {ckpt_path}")
            
            diff_wrapper.train()
            accelerator.print(f"Stem '{instr}': Diffusion model type '{diffusion_type}'")
            diffusion_wrappers.append(diff_wrapper)

        diffusion_wrappers = [accelerator.prepare(d) if d is not None else None for d in diffusion_wrappers]
    # End of Diffusion model setup

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        accelerator.print('Optimizer params from config:\n{}'.format(optim_params))

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'sgd':
        accelerator.print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        accelerator.print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    if accelerator.is_main_process:
        print('Processes GPU: {}'.format(accelerator.num_processes))
        print("Patience: {} Reduce factor: {} Batch size: {} Optimizer: {}".format(
            config.training.patience,
            config.training.reduce_factor,
            batch_size,
            config.training.optimizer,
        ))
    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        'max',
        # patience=accelerator.num_processes * config.training.patience, # This is strange place...
        patience=config.training.patience,
        factor=config.training.reduce_factor
    )

    if args.use_multistft_loss:
        try:
            loss_options = dict(config.loss_multistft)
        except Exception:
            loss_options = dict()
        accelerator.print('Loss options: {}'.format(loss_options))
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(
            **loss_options
        )

    # 修复4: 重新排列prepare顺序 - 先prepare model和optimizer
    # 这样可以确保model被正确地复制到各个GPU上
    model, optimizer = accelerator.prepare(model, optimizer)
    train_loader, scheduler = accelerator.prepare(train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)
    
    accelerator.print(f"[DEBUG] Model prepared on device: {next(model.parameters()).device}")

    def gather_sdr_metrics(accelerator, sdr_dict):
        result = {}
        for instr, values in sdr_dict.items():
            dtype = values[0].dtype if values else torch.float32
            local_tensor = torch.cat(values, dim=0) if values else torch.empty(0, device=accelerator.device, dtype=dtype)
            result[instr] = accelerator.gather_for_metrics(local_tensor).cpu()
        return result

    if args.pre_valid:
        metrics_dict = valid(model, valid_loader, args, config, device, verbose=accelerator.is_main_process)
        
        # Gather metrics from all processes
        gathered_metrics = {}
        for metric_name, metric_data in metrics_dict.items():
            gathered_metrics[metric_name] = gather_sdr_metrics(accelerator, metric_data)
        
        accelerator.wait_for_everyone()

        instruments = prefer_target_instrument(config)
        
        # 打印所有指标
        for metric_name in metric_list:
            metric_avg = 0.0
            for instr in instruments:
                metric_data = gathered_metrics[metric_name][instr].numpy()
                metric_val = metric_data[:valid_dataset_length].mean()
                accelerator.print("Instr {} {}: {:.4f}".format(instr, metric_name.upper(), metric_val))
                metric_avg += metric_val
            
            metric_avg /= len(instruments)
            if len(instruments) > 1:
                accelerator.print('{} Avg: {:.4f}'.format(metric_name.upper(), metric_avg))
        
        metrics_dict = None
        gathered_metrics = None

    accelerator.print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = 0
    best_si_sdr = -float('inf')  # 添加SI-SDR追踪
    current_step = 0
    for epoch in range(config.training.num_epochs):
        model.train()
        accelerator.print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, ncols=130)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch
            x = mixes

            if args.model_type in ['mel_band_roformer', 'bs_roformer', 'mel_band_roformer_disc', 'td_mel_band_roformer', 'band_conditioned_mel_band_roformer']:
                # loss is computed in forward pass
                loss, y_ = model(x, y)
            else:
                y_ = model(x)
                if args.use_multistft_loss:
                    y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
                    y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
                    loss = loss_multistft(y1_, y1)
                    # We can use many losses at the same time
                    if args.use_mse_loss:
                        loss += 1000 * nn.MSELoss()(y1_, y1)
                    if args.use_l1_loss:
                        loss += 1000 * F.l1_loss(y1_, y1)
                    if args.use_frequency_weighted_loss:
                        pass                    
                elif args.use_mse_loss:
                    loss = nn.MSELoss()(y_, y)
                elif args.use_l1_loss:
                    loss = F.l1_loss(y_, y)
                else:
                    loss = masked_loss(
                        y_,
                        y,
                        q=config.training.q,
                        coarse=config.training.coarse_loss_clip
                    )
                    
            # Discriminator training step - 支持多个判别器
            if args.use_discriminator:
                B, I, C, T = y_.shape
                num_stems = len(config.training.instruments)
                assert I == num_stems, f"Expected {num_stems} stems, got {I}"
                
                total_gan_loss = 0.0
                total_fm_loss = 0.0
                disc_losses_dict = {}
                gen_losses_dict = {}
                
                # 帧长度（仅mel类型需要）
                len_samples = torch.full((B,), T, device=y_.device, dtype=torch.long)
                len_frames = samp_to_frames(len_samples, C) if mel is not None else None
                
                for stem_idx, (disc, input_type, instr) in enumerate(zip(discriminators, gan_models, config.training.instruments)):
                    # 跳过没有判别器的stem
                    if disc is None or input_type is None or input_type.lower() == 'none':
                        continue
                    
                    # 提取当前stem的真假数据: [B, C, T]
                    fake_stem = y_[:, stem_idx, :, :]
                    real_stem = y[:, stem_idx, :, :]
                    
                    # 根据 input_type 准备输入
                    if input_type == 'mel':
                        fake_input = wave_BC_to_mel(fake_stem)
                        real_input = wave_BC_to_mel(real_stem)
                        length_input = len_frames
                    elif input_type in ['wave', 'music']:
                        # wave/music 判别器接收 [B*C, T] 格式
                        fake_input = wave_BC_to_wave_B(fake_stem)
                        real_input = wave_BC_to_wave_B(real_stem)
                        length_input = None  # wave判别器不需要长度信息
                    else:
                        raise ValueError(f"Unknown input_type: {input_type}")
                    
                    # 1) 更新判别器
                    d_losses = disc.train_step(fake_input, real_input, length_input, current_step=current_step)
                    disc_losses_dict[f'disc_{instr}'] = d_losses.get('disc_loss', 0.0)
                    
                    # 2) 冻结判别器，计算生成器损失
                    for p in disc.discriminator.parameters():
                        p.requires_grad = False
                    
                    g_losses = disc.get_generator_losses(fake_input, real_input, length_input, current_step=current_step)
                    
                    gan_loss = g_losses.get('gan_loss', 0.0)
                    fm_loss = g_losses.get('hdn_loss', 0.0)
                    
                    total_gan_loss += gan_loss
                    total_fm_loss += fm_loss
                    
                    gen_losses_dict[f'gen_{instr}_gan'] = gan_loss.item() if isinstance(gan_loss, torch.Tensor) else gan_loss
                    gen_losses_dict[f'gen_{instr}_fm'] = fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss
                    
                    # 3) 解冻判别器
                    for p in disc.discriminator.parameters():
                        p.requires_grad = True
                
                    # 累加到总损失
                    loss = loss + total_gan_loss + total_fm_loss

            if args.use_diffusion:
                B, I, C, T = y_.shape
                diffusion_losses_dict = {}
                num_stems = len(config.training.instruments)
                assert I == num_stems, f"Expected {num_stems} stems, got {I}"
                                
                for stem_idx, (diff_model, instr) in enumerate(zip(diffusion_wrappers, config.training.instruments)):
                    if diff_model is None:
                        continue
                    
                    # 提取当前stem的预测数据: [B, C, T]
                    pred_stem = y_[:, stem_idx, :, :]
                    target_stem = y[:, stem_idx, :, :]
                    
                    diffusion_loss = diff_model.train_step_dual(pred_stem, target_stem, current_step=current_step)
                    diffusion_losses_dict[f'diff_{instr}_loss'] = diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss
                
            accelerator.backward(loss)
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            
            li = loss.item()
            loss_val += li
            total += 1
            if accelerator.is_main_process:
                log_dict = {'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'total': total, 'loss_val': loss_val, 'i': i}
                
                # 添加所有判别器损失到日志
                if args.use_discriminator:
                    log_dict.update(disc_losses_dict)
                    log_dict.update(gen_losses_dict)
                    log_dict['total_gan_loss'] = total_gan_loss.item() if isinstance(total_gan_loss, torch.Tensor) else total_gan_loss
                    log_dict['total_fm_loss'] = total_fm_loss.item() if isinstance(total_fm_loss, torch.Tensor) else total_fm_loss
                
                # 添加所有扩散模型损失到日志
                if args.use_diffusion:
                    log_dict.update(diffusion_losses_dict)
                
                wandb.log(log_dict)
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
                
            # 仅保存最后的判别器检查点0
            if (args.use_discriminator or args.use_diffusion) and (current_step + 1) % getattr(config.training, 'save_interval', 1000) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(args.results_path, "last_additional_model_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                for disc, instr in zip(discriminators, config.training.instruments):
                    if disc is not None:  # 只保存非None的判别器
                        disc.save_checkpoint(os.path.join(ckpt_dir, f'{instr}.pth'))
                accelerator.print(f"Discriminator checkpoints saved to {ckpt_dir}")
                
                for diff_model, instr in zip(diffusion_wrappers, config.training.instruments):
                    if diff_model is not None:
                        diff_model.save_checkpoint(os.path.join(ckpt_dir, f'{instr}_diffusion.pth'))
                accelerator.print(f"Diffusion model checkpoints saved to {ckpt_dir}")
            
            current_step += 1

        if accelerator.is_main_process:
            print('Training loss: {:.6f}'.format(loss_val / total))
            wandb.log({'train_loss': loss_val / total, 'epoch': epoch})

        # Save last
        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), store_path)

        metrics_dict = valid(model, valid_loader, args, config, device, 
                             verbose=accelerator.is_main_process, diffusion_wrapper=diffusion_wrappers if args.use_diffusion else None, 
                             current_step=current_step)
        
        # Gather metrics from all processes
        gathered_metrics = {}
        for metric_name, metric_data in metrics_dict.items():
            gathered_metrics[metric_name] = gather_sdr_metrics(accelerator, metric_data)
        
        accelerator.wait_for_everyone()

        # 计算所有指标的平均值
        sdr_avg = 0.0
        si_sdr_avg = 0.0
        instruments = prefer_target_instrument(config)
        
        metrics_to_log = {}
        
        for metric_name in metric_list:
            metric_avg = 0.0
            for instr in instruments:
                if accelerator.is_main_process and 0:
                    print(gathered_metrics[metric_name][instr])
                metric_data = gathered_metrics[metric_name][instr].numpy()
                metric_val = metric_data[:valid_dataset_length].mean()
                
                if accelerator.is_main_process:
                    print("Instr {} {}: {:.4f} Debug: {}".format(instr, metric_name.upper(), metric_val, len(metric_data)))
                    metrics_to_log[f'{instr}_{metric_name}'] = metric_val
                
                metric_avg += metric_val
            
            metric_avg /= len(instruments)
            
            # 保存SDR和SI-SDR的平均值
            if metric_name == 'sdr':
                sdr_avg = metric_avg
            elif metric_name == 'si_sdr':
                si_sdr_avg = metric_avg
            
            if accelerator.is_main_process:
                metrics_to_log[f'{metric_name}_avg'] = metric_avg
                if len(instruments) > 1:
                    print('{} Avg: {:.4f}'.format(metric_name.upper(), metric_avg))
        
        # 记录所有指标到wandb
        if accelerator.is_main_process:
            metrics_to_log['best_sdr'] = best_sdr
            metrics_to_log['best_si_sdr'] = best_si_sdr
            metrics_to_log['epoch'] = epoch
            wandb.log(metrics_to_log)

        # 使用SI-SDR保存最佳检查点
        if accelerator.is_main_process:
            if si_sdr_avg > best_si_sdr:
                store_path = args.results_path + '/model_{}_ep_{}_sisdr_{:.4f}.ckpt'.format(args.model_type, epoch, si_sdr_avg)
                print('Store weights (Best SI-SDR): {}'.format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_si_sdr = si_sdr_avg
            
            # 同时记录SDR最佳（可选）
            if sdr_avg > best_sdr:
                best_sdr = sdr_avg

            # 使用SI-SDR作为调度器的指标
            scheduler.step(si_sdr_avg)

        metrics_dict = None
        gathered_metrics = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    train_model(None)
