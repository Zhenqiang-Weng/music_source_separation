# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.6'

import argparse
import soundfile as sf
import numpy as np
import time
import glob
from tqdm.auto import tqdm
import os
import torch
import wandb
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from utils.dataset_with_spk import MSSDatasetWithSpk
from utils.model_utils import prefer_target_instrument, demix_with_spk
from utils.metrics import get_metrics
from utils.settings import manual_seed, get_model_from_config
from utils.losses import masked_loss

import warnings
warnings.filterwarnings("ignore")

from models.preprocess.feature_extract import STFTProcessorBatch, STFTConfig
from discriminator.discriminator_wrapper import DiscriminatorWrapper, DiscriminatorConfig

METRIC_LIST = ["sdr", "si_sdr"]
AVAILABLE_MODELS = ['bs_roformer_exportable', 'spk_bs_roformer_exportable', 'speaker_mel_band_roformer_exportable']

def _get_windowing_array(window_size, fade_size, device):
    window = torch.ones(window_size, device=device)
    fadein = torch.linspace(0, 1, fade_size, device=device)
    fadeout = torch.linspace(1, 0, fade_size, device=device)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


def demix_with_spk_exportable(config, model, mix, spk_emb, device, stft_processor):
    model.eval()

    if hasattr(config, 'inference') and hasattr(config.inference, 'chunk_size'):
        chunk_size = config.inference.chunk_size
    else:
        chunk_size = getattr(config.audio, 'chunk_size', 882000)

    num_overlap = getattr(config.inference, 'num_overlap', 2)
    step = chunk_size // num_overlap
    border = chunk_size - step
    fade_size = step // 10

    instruments = config.training.instruments
    num_stems = len(instruments)

    if isinstance(mix, np.ndarray):
        mix = torch.from_numpy(mix).float()
    mix = mix.to(device)

    if mix.ndim == 1:
        mix = mix.unsqueeze(0).repeat(2, 1)
    elif mix.shape[0] == 1:
        mix = mix.repeat(2, 1)

    channels, length_init = mix.shape

    if length_init > 2 * border and border > 0:
        mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")
    else:
        border = 0

    total_samples = mix.shape[1]
    result = torch.zeros((num_stems, channels, total_samples), device=device)
    counter = torch.zeros((1, 1, total_samples), device=device)

    windowing_array = _get_windowing_array(chunk_size, fade_size, device)

    start = 0
    with torch.no_grad():
        while start < total_samples:
            end = start + chunk_size
            if end > total_samples:
                end = total_samples

            chunk = mix[:, start:end]
            actual_len = chunk.shape[1]
            if actual_len < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - actual_len), mode="constant", value=0)

            chunk_input = chunk.unsqueeze(0)
            chunk_stft = stft_processor.stft_batch(chunk_input)
            output_stft = model(chunk_stft, spk_emb)
            output_audio = stft_processor.istft_batch(output_stft).squeeze(0)

            window = windowing_array.clone()
            if start == 0:
                window[:fade_size] = 1.0
            if start + chunk_size >= total_samples:
                window[-fade_size:] = 1.0

            output_audio = output_audio[:, :, :actual_len]
            win_part = window[:actual_len]

            result[:, :, start:end] += output_audio * win_part
            counter[:, :, start:end] += win_part

            start += step

    result = result / torch.clamp(counter, min=1e-8)

    if border > 0:
        result = result[:, :, border:-border]

    result_np = result.cpu().numpy()

    output = {}
    for i, instr in enumerate(instruments):
        output[instr] = result_np[i]

    return output


def valid(model, valid_loader, args, config, device, verbose=False, stft_processor=None):
    instruments = prefer_target_instrument(config)

    all_metrics = {}
    for metric in METRIC_LIST:
        all_metrics[metric] = {}
        for instr in instruments:
            all_metrics[metric][instr] = []

    all_mixtures_path = valid_loader
    if verbose:
        all_mixtures_path = tqdm(valid_loader, ncols=200)

    pbar_dict = {}
    for batch in all_mixtures_path:
        path = batch[0][0]
        spk_emb = batch[1].to(device)

        mix, sr = sf.read(path)
        folder = os.path.dirname(path)

        if args.model_type in AVAILABLE_MODELS:
            res = demix_with_spk_exportable(config, model, mix.T, spk_emb, device, stft_processor)
        else:
            res = demix_with_spk(config, model, mix.T, spk_emb, device, model_type=args.model_type)

        for instr in instruments:
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
            else:
                track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                track = mix - track
            references = np.expand_dims(track, axis=0)
            estimates = np.expand_dims(res[instr].T, axis=0)

            results = get_metrics(METRIC_LIST, references, estimates, mix)
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
        path = self.list_of_files[index]
        folder = os.path.dirname(path)
        embedding_path = os.path.join(folder, 'embeddings.npy')

        embedding = torch.zeros(192)

        if os.path.exists(embedding_path):
            try:
                embeddings = np.load(embedding_path)
                if embeddings.shape[0] > 0:
                    emb_mean = np.mean(embeddings, axis=0)
                    embedding = torch.from_numpy(emb_mean).float()
            except Exception as e:
                print(f"Error loading embedding for {path}: {e}")

        return path, embedding


def train_model(args):
    accelerator = Accelerator()
    device = accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c')
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--start_check_point", type=str, default='')
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--data_path", nargs="+", type=str)
    parser.add_argument("--dataset_type", type=int, default=1)
    parser.add_argument("--valid_path", nargs="+", type=str)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument("--use_multistft_loss", action='store_true')
    parser.add_argument("--use_mse_loss", action='store_true')
    parser.add_argument("--use_l1_loss", action='store_true')
    parser.add_argument("--wandb_key", type=str, default='')
    parser.add_argument("--pre_valid", action='store_true')
    parser.add_argument("--wandb_project", type=str, default='msst-accelerate')
    parser.add_argument("--wandb_name", type=str, default='')
    parser.add_argument("--use_mix_consistent_loss", action='store_true')
    parser.add_argument("--use_discriminator", action='store_true', help="Use discriminator for training")
    parser.add_argument("--discriminator_start_check_point", type=str, default='', help="Initial checkpoint to start discriminator training")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    manual_seed(args.seed + int(time.time()))
    torch.backends.cudnn.deterministic = False

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        accelerator.print(f"Warning: multiprocessing start method already set: {e}")

    model, config = get_model_from_config(args.model_type, args.config_path)
    accelerator.print("Instruments: {}".format(config.training.instruments))
    accelerator.print(f"[DEBUG] Number of processes: {accelerator.num_processes}")
    accelerator.print(f"[DEBUG] Device: {accelerator.device}")
    accelerator.print(f"[DEBUG] Is main process: {accelerator.is_main_process}")

    os.makedirs(args.results_path, exist_ok=True)

    device_ids = args.device_ids
    batch_size = config.training.batch_size

    if accelerator.is_main_process:
        if args.wandb_key is not None and args.wandb_key.strip() != '':
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project, name=args.wandb_name, config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size})
        else:
            wandb.init(project=args.wandb_project, name=args.wandb_name, mode='offline', config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size})
    else:
        wandb.init(mode='disabled')

    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDatasetWithSpk(
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
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False)

    if args.start_check_point != '':
        accelerator.print('Start from checkpoint: {}'.format(args.start_check_point))
        checkpoint = torch.load(args.start_check_point, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        accelerator.print("Checkpoint loaded successfully")

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

    scheduler = ReduceLROnPlateau(
        optimizer, 'max',
        patience=config.training.patience,
        factor=config.training.reduce_factor
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    train_loader, scheduler = accelerator.prepare(train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)

    accelerator.print(f"[DEBUG] Model prepared on device: {next(model.parameters()).device}")

    stft_cfg = STFTConfig(
        n_fft=getattr(config.audio, 'n_fft', 2048),
        hop_length=getattr(config.audio, 'hop_length', 512),
        win_length=getattr(config.audio, 'stft_win_length', 2048),
        normalized=getattr(config.model, 'stft_normalized', False),
        center=True,
        istft_length=getattr(config.audio, 'chunk_size', None),
        stereo=getattr(config.model, 'stereo', True),
        num_stems=getattr(config.model, 'num_stems', 1),
        use_multi_stft_loss=True,
        multi_weight=getattr(config.model, 'multi_stft_resolution_loss_weight', 1.0),
        multi_window_sizes=tuple(getattr(config.model, 'multi_stft_resolutions_window_sizes', (4096, 2048, 1024, 512, 256))),
        multi_hop_length=getattr(config.model, 'multi_stft_hop_size', 147),
        multi_normalized=getattr(config.model, 'multi_stft_normalized', False),
    )
    stft_processor = STFTProcessorBatch(stft_cfg, device=device)

    discriminators = []
    if args.use_discriminator:
        gan_models = getattr(config.training, 'gan_model', None)
        num_stems = len(config.training.instruments)
        
        if gan_models is None:
            accelerator.print("Warning: gan_model not found in config, using default 'wave' for all stems")
            gan_models = ['wave'] * num_stems
        elif len(gan_models) != num_stems:
            raise ValueError(f"gan_model length ({len(gan_models)}) must match number of instruments ({num_stems})")
        
        accelerator.print(f"Creating discriminators for {num_stems} stems with types: {gan_models}")
        
        for idx, (instr, gan_type) in enumerate(zip(config.training.instruments, gan_models)):
            if gan_type in [None, 'none']:
                discriminators.append(None)
                accelerator.print(f"Discriminator for stem {idx} ({instr}): DISABLED")
            else:
                disc_cfg = DiscriminatorConfig(input_type=gan_type)
                disc_wrapper = DiscriminatorWrapper(disc_cfg, accelerator)
                
                if args.discriminator_start_check_point != '':
                    ckpt_path = os.path.join(args.discriminator_start_check_point, f'{instr}.pth')
                    if os.path.exists(ckpt_path):
                        disc_wrapper.load_checkpoint(ckpt_path)
                        accelerator.print(f"Loaded discriminator checkpoint for {instr} from {ckpt_path}")
                
                disc_wrapper.train()
                discriminators.append(disc_wrapper)
                accelerator.print(f"Discriminator for stem {idx} ({instr}): type '{gan_type}'")

    def gather_sdr_metrics(accelerator, sdr_dict):
        result = {}
        for instr, values in sdr_dict.items():
            dtype = values[0].dtype if values else torch.float32
            local_tensor = torch.cat(values, dim=0) if values else torch.empty(0, device=accelerator.device, dtype=dtype)
            result[instr] = accelerator.gather_for_metrics(local_tensor).cpu()
        return result

    if args.pre_valid:
        sdr_list = valid(model, valid_loader, args, config, device, verbose=accelerator.is_main_process, stft_processor=stft_processor)
        sdr_list = gather_sdr_metrics(accelerator, sdr_list['sdr'])
        accelerator.wait_for_everyone()
        sdr_avg = 0.0
        instruments = prefer_target_instrument(config)
        for instr in instruments:
            sdr_data = sdr_list[instr].numpy()
            sdr_val = sdr_data[:valid_dataset_length].mean()
            accelerator.print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
            sdr_avg += sdr_val
        sdr_avg /= len(instruments)
        if len(instruments) > 1:
            accelerator.print('SDR Avg: {:.4f}'.format(sdr_avg))
        sdr_list = None

    accelerator.print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = 0
    best_si_sdr = -float('inf')
    current_step = 0

    for epoch in range(config.training.num_epochs):
        model.train()
        accelerator.print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, ncols=130)
        for i, (batch, mixes, spk_emb) in enumerate(pbar):
            y = batch
            x = mixes

            if args.model_type in ['bs_roformer_exportable', 'spk_bs_roformer_exportable', 'speaker_mel_band_roformer_exportable']:
                x_stft = stft_processor.stft_batch(x)
                y_stft = model(x_stft, spk_emb)
                y_ = stft_processor.istft_batch(y_stft)
                use_multi_stft = getattr(config.training, 'use_multi_stft_loss', True)
                loss = stft_processor.compute_loss(y_, y, use_multi_stft=use_multi_stft)
            else:
                loss, y_ = model(x, spk_emb, y)

            if args.use_mix_consistent_loss:
                all_ins = y_.sum(dim=1)
                pre_all_ins = y.sum(dim=1)
                total_mix_loss = F.l1_loss(all_ins, pre_all_ins)
                total_mix_consistent_loss_weight = getattr(config.training, 'total_mix_consistent_loss_weight', 0.5)
                loss += total_mix_consistent_loss_weight * total_mix_loss
                mixing_loss_value = total_mix_loss.item()

            if args.use_discriminator:
                B, I, C, T = y_.shape
                num_stems = len(config.training.instruments)
                assert I == num_stems, f"Expected {num_stems} stems, got {I}"
                
                total_gan_loss = 0.0
                total_fm_loss = 0.0
                disc_losses_dict = {}
                gen_losses_dict = {}
                
                for stem_idx in range(num_stems):
                    disc = discriminators[stem_idx]
                    if disc is None:
                        continue
                    
                    instr = config.training.instruments[stem_idx]
                    fake_audio = y_[:, stem_idx, :, :]
                    real_audio = y[:, stem_idx, :, :]
                    
                    fake_input = fake_audio.reshape(B * C, T)
                    real_input = real_audio.reshape(B * C, T)
                    
                    d_losses = disc.train_step(fake_input, real_input, None, current_step=current_step)
                    disc_losses_dict[f'disc_{instr}'] = d_losses.get('disc_loss', 0.0)
                    
                    for p in disc.discriminator.parameters():
                        p.requires_grad = False
                    
                    g_losses = disc.get_generator_losses(fake_input, real_input, None, current_step=current_step)
                    gan_loss = g_losses.get('gan_loss', 0.0)
                    fm_loss = g_losses.get('hdn_loss', 0.0)
                    
                    total_gan_loss += gan_loss
                    total_fm_loss += fm_loss
                    
                    gen_losses_dict[f'gen_{instr}_gan'] = gan_loss.item() if isinstance(gan_loss, torch.Tensor) else gan_loss
                    gen_losses_dict[f'gen_{instr}_fm'] = fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss
                    
                    for p in disc.discriminator.parameters():
                        p.requires_grad = True
                
                loss = loss + total_gan_loss + total_fm_loss

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
                if args.use_mix_consistent_loss:
                    log_dict['mix_consistent_loss'] = 100 * mixing_loss_value
                
                if args.use_discriminator:
                    log_dict.update(disc_losses_dict)
                    log_dict.update(gen_losses_dict)
                    log_dict['total_gan_loss'] = total_gan_loss.item() if isinstance(total_gan_loss, torch.Tensor) else total_gan_loss
                    log_dict['total_fm_loss'] = total_fm_loss.item() if isinstance(total_fm_loss, torch.Tensor) else total_fm_loss
                
                wandb.log(log_dict)
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            
            if args.use_discriminator and (current_step + 1) % getattr(config.training, 'save_interval', 1000) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(args.results_path, "last_discriminator_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                for disc, instr in zip(discriminators, config.training.instruments):
                    if disc is not None:
                        disc.save_checkpoint(os.path.join(ckpt_dir, f'{instr}.pth'))
                accelerator.print(f"Discriminator checkpoints saved to {ckpt_dir}")

            current_step += 1

        if accelerator.is_main_process:
            print('Training loss: {:.6f}'.format(loss_val / total))
            wandb.log({'train_loss': loss_val / total, 'epoch': epoch})

        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), store_path)

        metrics_dict = valid(model, valid_loader, args, config, device, verbose=accelerator.is_main_process, stft_processor=stft_processor)

        gathered_metrics = {}
        for metric_name, metric_data in metrics_dict.items():
            gathered_metrics[metric_name] = gather_sdr_metrics(accelerator, metric_data)

        accelerator.wait_for_everyone()

        sdr_avg = 0.0
        si_sdr_avg = 0.0
        instruments = prefer_target_instrument(config)
        metrics_to_log = {}

        for metric_name in METRIC_LIST:
            metric_avg = 0.0
            for instr in instruments:
                metric_data = gathered_metrics[metric_name][instr].numpy()
                metric_val = metric_data[:valid_dataset_length].mean()
                if accelerator.is_main_process:
                    print("Instr {} {}: {:.4f} Debug: {}".format(instr, metric_name.upper(), metric_val, len(metric_data)))
                    metrics_to_log[f'{instr}_{metric_name}'] = metric_val
                metric_avg += metric_val
            metric_avg /= len(instruments)
            if metric_name == 'sdr':
                sdr_avg = metric_avg
            elif metric_name == 'si_sdr':
                si_sdr_avg = metric_avg
            if accelerator.is_main_process:
                metrics_to_log[f'{metric_name}_avg'] = metric_avg
                if len(instruments) > 1:
                    print('{} Avg: {:.4f}'.format(metric_name.upper(), metric_avg))

        if accelerator.is_main_process:
            metrics_to_log['best_sdr'] = best_sdr
            metrics_to_log['best_si_sdr'] = best_si_sdr
            metrics_to_log['epoch'] = epoch
            wandb.log(metrics_to_log)

            if si_sdr_avg > best_si_sdr:
                store_path = args.results_path + '/model_{}_ep_{}_sisdr_{:.4f}.ckpt'.format(args.model_type, epoch, si_sdr_avg)
                print('Store weights (Best SI-SDR): {}'.format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_si_sdr = si_sdr_avg

            if sdr_avg > best_sdr:
                store_path = args.results_path + '/model_{}_ep_{}_sdr_{:.4f}.ckpt'.format(args.model_type, epoch, sdr_avg)
                print('Store weights (Best SDR): {}'.format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_sdr = sdr_avg

            scheduler.step(sdr_avg)

        metrics_dict = None
        gathered_metrics = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    train_model(None)