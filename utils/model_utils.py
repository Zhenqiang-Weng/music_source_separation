import argparse
import json
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union, Optional
import loralib as lora
from .muon import SingleDeviceMuonWithAuxAdam
import torch.distributed as dist
from spk_extract import extract_dominant_speaker_embedding_with_clusters

def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform audio source separation with a given model.

    Supports both Demucs-specific and generic processing modes, including
    overlapping chunk-based inference with optional progress bar display.
    Handles padding, fading, and batching to reduce artifacts during separation.

    Args:
        config (ConfigDict): Configuration object with audio and inference
            parameters (chunk size, overlap, batch size, etc.).
        model (torch.nn.Module): Source separation model for inference.
        mix (torch.Tensor): Input audio tensor of shape (channels, time).
        device (torch.device): Device on which to run inference (CPU or CUDA).
        model_type (str): Type of model (e.g., 'htdemucs', 'mdx23c') that
            determines processing mode.
        pbar (bool, optional): If True, show a progress bar during chunk
            processing. Defaults to False.

    Returns:
        Union[Dict[str, np.ndarray], np.ndarray]:
            - Dictionary mapping instrument names to separated waveforms if
              multiple instruments are predicted.
            - NumPy array of separated audio if only a single instrument is
              present (Demucs mode).
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        if 'chunk_size' in config.inference:
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            if pbar and should_print:
                progress_bar = tqdm(
                    total=mix.shape[1], desc="Processing audio chunks", leave=False
                )
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def demix_with_spk(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    spk_emb: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform audio source separation with a given model and speaker embedding.
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        if 'chunk_size' in config.inference:
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            if pbar and should_print:
                progress_bar = tqdm(
                    total=mix.shape[1], desc="Processing audio chunks", leave=False
                )
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    
                    # Prepare speaker embedding
                    current_bs = arr.shape[0]
                    if spk_emb.dim() == 1:
                        batch_spk = spk_emb.unsqueeze(0).repeat(current_bs, 1).to(device)
                    else:
                        batch_spk = spk_emb.repeat(current_bs, 1).to(device)
                    
                    x = model(arr, batch_spk)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)
            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data



def demix_with_spk_2(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    spk_model: torch.nn.Module,
    device: torch.device,
    model_type: str,
    pbar: bool = False,
    segment_duration: float = 2.0,
    energy_threshold: float = 0.1,
    max_clusters: int = 3,
    debug: bool = False,
    apply_gate: bool = False,
    gate_interval: float = 0.1,
    gate_threshold: float = 0.01
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    带说话人聚类的音频分离：先盲分离，再从vocals提取embedding进行引导分离
    
    Pipeline:
    1. 先进行一次盲分离 (demix)，得到初步的 vocals
    2. (可选) 对 vocals 应用能量门控
    3. 从分离出的 vocals 中提取 speaker embedding，进行谱聚类
    4. 找到最大簇（主导说话人），计算其平均 speaker embedding
    5. 使用该 embedding 对整个音频重新进行分离 (demix_with_spk)
    6. (可选) 对最终结果应用能量门控
    
    Args:
        config: 配置对象
        model: 分离模型
        mix: 输入混合音频 (channels, samples) @ 44100Hz
        spk_model: 说话人识别模型 (CAMPPlus)
        device: 计算设备
        model_type: 模型类型
        pbar: 是否显示进度条
        segment_duration: 说话人聚类的片段时长 (秒)
        energy_threshold: 聚类时的能量阈值
        max_clusters: 最大聚类数
        debug: 调试模式
        apply_gate: 是否应用能量门控 (default: False)
        gate_interval: 门控检测间隔，秒 (default: 0.1s)
        gate_threshold: 门控能量阈值 (default: 0.01)
    
    Returns:
        分离结果字典 {instrument: waveform}
    """
    
    should_print = not dist.is_initialized() or dist.get_rank() == 0
    
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    
    # =========================================================================
    # Step 1: 先进行一次盲分离，得到初步的 vocals
    # =========================================================================
    if debug and should_print:
        print(f"[demix_with_spk_2] Step 1: 盲分离，获取初步 vocals...")
    
    waveforms_blind = demix(config, model, mix, device, model_type=model_type, pbar=pbar)
    
    # Step 1.5: (可选) 对盲分离结果应用能量门控
    if apply_gate:
        if debug and should_print:
            print(f"[demix_with_spk_2] 应用能量门控 (interval={gate_interval}s, threshold={gate_threshold})...")
        waveforms_blind = apply_energy_gate(
            waveforms_blind,
            sample_rate=sample_rate,
            gate_interval=gate_interval,
            energy_threshold=gate_threshold,
            target_instrument='vocals'
        )
    
    # 获取 vocals (用于提取 speaker embedding)
    if 'vocals' in waveforms_blind:
        vocals_for_embedding = waveforms_blind['vocals']
    else:
        # 如果没有 vocals，使用第一个 instrument
        first_instr = list(waveforms_blind.keys())[0]
        vocals_for_embedding = waveforms_blind[first_instr]
        if debug and should_print:
            print(f"[demix_with_spk_2] 警告: 没有找到 vocals，使用 {first_instr} 提取 embedding")
    
    if debug and should_print:
        print(f"[demix_with_spk_2] 盲分离完成，vocals shape: {vocals_for_embedding.shape}")
    
    # =========================================================================
    # Step 2: 从分离出的 vocals 中提取 speaker embedding 并聚类
    # =========================================================================
    if debug and should_print:
        print(f"[demix_with_spk_2] Step 2: 从 vocals 提取 speaker embedding 并聚类...")
    
    clustering_result = extract_dominant_speaker_embedding_with_clusters(
        model=spk_model,
        audio=vocals_for_embedding,  # 使用分离出的 vocals
        source_sr=sample_rate,
        target_sr=16000,
        segment_duration=segment_duration,
        energy_threshold=energy_threshold,
        max_clusters=max_clusters,
        device=device,
        debug=debug
    )
    
    if clustering_result is None:
        if should_print:
            print("[demix_with_spk_2] 聚类失败，直接返回盲分离结果")
        return waveforms_blind
    
    # 获取最大簇的 speaker embedding
    dominant_spk_emb = clustering_result.mean_embedding.to(device)
    
    if debug and should_print:
        print(f"[demix_with_spk_2] 最大簇标签: {clustering_result.largest_cluster_label}")
        print(f"[demix_with_spk_2] 各片段标签: {clustering_result.labels.tolist()}")
        print(f"[demix_with_spk_2] 主导说话人 embedding shape: {dominant_spk_emb.shape}")
    
    # =========================================================================
    # Step 3: 使用主导说话人 embedding 重新进行完整分离
    # =========================================================================
    if debug and should_print:
        print(f"[demix_with_spk_2] Step 3: 使用主导说话人 embedding 重新分离整首歌...")
    
    waveforms = demix_with_spk(
        config, model, mix, dominant_spk_emb, device, 
        model_type=model_type, pbar=pbar
    )
    
    # Step 3.5: (可选) 对最终结果应用能量门控
    if apply_gate:
        if debug and should_print:
            print(f"[demix_with_spk_2] 对最终结果应用能量门控...")
        waveforms = apply_energy_gate(
            waveforms,
            sample_rate=sample_rate,
            gate_interval=gate_interval,
            energy_threshold=gate_threshold,
            target_instrument='vocals'
        )
    
    if debug and should_print:
        print(f"[demix_with_spk_2] 完成!")
    
    return waveforms

def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    Move a model to the correct computation device and wrap with DataParallel if needed.

    Selects GPU(s) if CUDA is available; otherwise defaults to CPU. If multiple
    GPU IDs are provided, wraps the model with `nn.DataParallel` for multi-GPU
    execution.

    Args:
        model (torch.nn.Module): PyTorch model to be initialized.
        device_ids (List[int]): List of GPU device IDs to use. If length > 1,
            the model will be wrapped with DataParallel.

    Returns:
        Tuple[Union[torch.device, str], torch.nn.Module]: A tuple containing:
            - The computation device (`torch.device` or "cpu").
            - The model moved to that device (wrapped in DataParallel if applicable).
    """

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Create and configure an optimizer for training.

    Selects the optimizer type based on `config.training.optimizer` and applies
    the corresponding parameters, including support for advanced optimizers
    such as Muon, Prodigy, and 8-bit AdamW. Handles parameter group separation
    for specialized optimizers (e.g., Muon vs. Adam parameters).

    Args:
        config (ConfigDict): Training configuration containing optimizer type,
            learning rate, and optional optimizer-specific parameters.
        model (torch.nn.Module): Model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Initialized optimizer ready for training.

    Raises:
        ValueError: If required optimizer configuration is missing (e.g., for Muon).
        SystemExit: If an unknown optimizer name is encountered.
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0
    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        if config.training.optimizer != 'muon' and should_print:
            print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'muon':
        if should_print:
            print("Using Muon optimizer (Single-Device) with AdamW for auxiliary parameters.")
        
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.ndim < 2]

        if not hasattr(config, 'optimizer') or 'muon_group' not in config.optimizer or 'adam_group' not in config.optimizer:
            raise ValueError("For the 'muon' optimizer, the config must have an 'optimizer' section "
                             "with 'muon_group' and 'adam_group' dictionaries.")

        muon_group_config = dict(config.optimizer.muon_group)
        adam_group_config = dict(config.optimizer.adam_group)

        if should_print:
            print(f"Muon group params: {muon_group_config}")
            print(f"Adam group params: {adam_group_config}")

        param_groups = [
            dict(params=muon_params, use_muon=True, **muon_group_config),
            dict(params=adam_params, use_muon=False, **adam_group_config),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    elif name_optimizer == 'sgd':
        if should_print:
            print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        if should_print:
            print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mean-variance normalization to a pair of tensors.

    Computes the mean and standard deviation from `x` and normalizes both `x`
    and `y` using those statistics. This ensures the two tensors are scaled
    consistently.

    Args:
        x (torch.Tensor): Input tensor used to compute normalization statistics.
        y (torch.Tensor): Input tensor normalized using the same statistics as `x`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Normalized tensors `(x, y)`.
    """

    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y


def apply_tta(
    config,
    model: torch.nn.Module,
    mix: torch.Tensor,
    waveforms_orig: Dict[str, torch.Tensor],
    device: torch.device,
    model_type: str
) -> Dict[str, torch.Tensor]:
    """
    Enhance source separation results using Test-Time Augmentation (TTA).

    Applies augmentations such as channel reversal and polarity inversion to
    the input mixture, reprocesses with the model, and combines the results
    with the original predictions by averaging.

    Args:
        config: Configuration object with model and inference parameters.
        model (torch.nn.Module): Trained source separation model.
        mix (torch.Tensor): Input mixture tensor of shape (channels, time).
        waveforms_orig (Dict[str, torch.Tensor]): Dictionary of separated
            sources before augmentation.
        device (torch.device): Computation device (CPU or CUDA).
        model_type (str): Model type identifier used for demixing.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of separated sources after applying TTA.
    """

    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
        Return the list of target instruments based on the configuration.
        If a specific target instrument is specified in the configuration,
        it returns a list with that instrument. Otherwise, it returns the list of instruments.

        Parameters:
        ----------
        config : ConfigDict
            Configuration object containing the list of instruments or the target instrument.

        Returns:
        -------
        List[str]
            A list of target instruments.
        """
    if getattr(config.training, 'target_instrument', None):
        if isinstance(config.training.target_instrument, str):
            return [config.training.target_instrument]
        elif isinstance(config.training.target_instrument, list):
            return config.training.target_instrument
    else:
        return config.training.instruments


def load_not_compatible_weights(model: torch.nn.Module, old_model: dict, verbose: bool = False) -> None:
    """
    Load a possibly incompatible state dict into `model` with best-effort matching.

    Accepts either a raw state_dict or a checkpoint dict with weights under "state" or "state_dict".
    For each param/buffer in `model`: if the name exists and shapes match → copy;
    if ndim matches but shapes differ → zero-pad/crop the source to fit the target;
    if the name is missing or ndim differs → skip. Optional logging on rank 0 when `verbose=True`.

    Args:
        model: Target PyTorch module.
        old_model: Source weights (state_dict or checkpoint dict).
        verbose: Print brief load decisions.

    Returns:
        None
    """

    should_print = verbose and (not dist.is_initialized() or dist.get_rank() == 0)

    new_model = model.state_dict()

    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # Fix for apollo weights loading
        old_model = old_model['state_dict']
    if 'model_state_dict' in old_model:
        # Fix for full_check_point
        old_model = old_model['model_state_dict']

    for el in new_model:
        if el in old_model:
            if should_print:
                print(f'Match found for {el}!')
            if new_model[el].shape == old_model[el].shape:
                if should_print:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape) and should_print:
                    print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if should_print:
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if should_print:
                print(f'Match not found for {el}!')
    model.load_state_dict(
        new_model
    )


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    Load LoRA weights into a model.
    This function updates the given model with LoRA-specific weights from the specified checkpoint file.
    It does not require the checkpoint to match the model's full state dictionary, as only LoRA layers are updated.

    Parameters:
    ----------
    model : Module
        The PyTorch model into which the LoRA weights will be loaded.
    lora_path : str
        Path to the LoRA checkpoint file.
    device : str, optional
        The device to load the weights onto, by default 'cpu'. Common values are 'cpu' or 'cuda'.

    Returns:
    -------
    None
        The model is updated in place.
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)


def load_start_checkpoint(args: argparse.Namespace,
                          model: torch.nn.Module,
                          old_model,
                          type_: str = 'train') -> None:
    """
    Load an initial checkpoint into `model`.

    For `type_ == "train"`, performs a tolerant load using `old_model` (a state dict or a
    checkpoint dict) via `load_not_compatible_weights`, allowing partial shape mismatches.
    For other modes, loads a strict state dict from `args.start_check_point`, with special
    handling for HTDemucs/Apollo checkpoints (keys under "state"/"state_dict"). If
    `args.lora_checkpoint` is set, LoRA weights are applied after the base load.

    Args:
        args: Namespace with at least `start_check_point`, `model_type`, and optionally `lora_checkpoint`.
        model: Target PyTorch module to receive weights.
        old_model: Source weights for tolerant loading in train mode (state dict or checkpoint dict).
        type_: Loading strategy; "train" uses tolerant loading, otherwise strict loading from path.

    Returns:
        None
    """
    should_print = not dist.is_initialized() or dist.get_rank() == 0

    if should_print:
        print(f'Start from checkpoint: {args.start_check_point}')
    if type_ in ['train']:
        if 1:
            load_not_compatible_weights(model, old_model, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))
    else:
        device='cpu'
        if args.model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            if 'state' in old_model:
                # Fix for htdemucs weights loading
                old_model = old_model['state']
            if 'state_dict' in old_model:
                # Fix for apollo weights loading
                old_model = old_model['state_dict']
            if 'model_state_dict' in old_model:
                # Fix for full_check_point
                old_model = old_model['model_state_dict']
        model.load_state_dict(old_model)

    if args.lora_checkpoint:
        if should_print:
            print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        load_lora_weights(model, args.lora_checkpoint)


def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    """
    Replaces specific layers in the model with LoRA-extended versions.

    Parameters:
    ----------
    config : Dict[str, Any]
        Configuration containing parameters for LoRA. It should include a 'lora' key with parameters for `MergedLinear`.
    model : nn.Module
        The original model in which the layers will be replaced.

    Returns:
    -------
    nn.Module
        The modified model with the replaced layers.
    """

    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    replaced_layers = 0  # Counter for replaced layers
    should_print = not dist.is_initialized() or dist.get_rank() == 0

    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]

        # Check if this is the target layer to replace (and layer_name == 'to_qkv')
        if isinstance(module, nn.Linear):
            try:
                # Get the parent module
                parent_module = model
                for submodule_name in hierarchy[:-1]:
                    parent_module = getattr(parent_module, submodule_name)

                # Replace the module with LoRA-enabled layer
                setattr(
                    parent_module,
                    layer_name,
                    lora.MergedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        **config['lora']
                    )
                )
                replaced_layers += 1  # Increment the counter

            except Exception as e:
                if should_print:
                    print(f"Error replacing layer {name}: {e}")

    if replaced_layers == 0 and should_print:
        print("Warning: No layers were replaced. Check the model structure and configuration.")
    elif should_print:
        print(f"Number of layers replaced with LoRA: {replaced_layers}")

    return model

def log_model_info(model: torch.nn.Module, results_path):
    """Log comprehensive model information"""
    model_info = {
        "timestamp": datetime.now().isoformat(),
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info["parameters"] = {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_millions": round(total_params / 1e6, 2),
        "trainable_millions": round(trainable_params / 1e6, 2),
    }

    # Get model size in memory
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    model_info["memory"] = {
        "parameters_mb": round(param_size / 1024 / 1024, 2),
        "buffers_mb": round(buffer_size / 1024 / 1024, 2),
        "total_mb": round(model_size_mb, 2),
    }

    # Log layer information
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                layer_info.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": layer_params,
                })

    model_info["layers"] = layer_info

    if results_path:
        path = os.path.join(results_path, "model_info.json")
        # Save model info
        with open(path, 'w') as f:
            json.dump(model_info, f, indent=2)

    # Log summary
    if not dist.is_initialized() or dist.get_rank()==0:
        print(f"Model: {model_info['model_class']}")
        print(f"Total parameters: {model_info['parameters']['total']:,} ({model_info['parameters']['total_millions']}M)")
        print(
            f"Trainable parameters: {model_info['parameters']['trainable']:,} ({model_info['parameters']['trainable_millions']}M)")
        print(f"Model size: {model_info['memory']['total_mb']:.2f} MB")
        print(f"Number of layers: {len(layer_info)}")


def save_weights(
    store_path: str,
    model: nn.Module,
    device_ids: List[int],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    all_time_all_metrics,
    best_metric: float,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    train_lora: bool = False
) -> None:
    """
    Save a training checkpoint containing model weights, optimizer/scheduler states, and metadata.

    Behavior:
    - In Distributed Data Parallel (DDP), only rank 0 writes the file to avoid conflicts.
    - If `train_lora` is True, saves only LoRA adapter weights (`lora_state_dict`); otherwise saves the full model.
    - Uses `model.module.state_dict()` when the model is wrapped by DDP/DataParallel.
    - Stores `epoch` and `best_metric` alongside optimizer/scheduler states.

    Args:
        store_path: Destination file path for the checkpoint (will be overwritten).
        model: The model whose weights are being saved (may be wrapped by DDP/DataParallel).
        device_ids: List of GPU device IDs used during training (used to detect DP wrapping in non-DDP runs).
        optimizer: Optimizer whose state will be saved.
        epoch: Current training epoch to record in the checkpoint.
        all_time_all_metrics:
        best_metric: Best validation metric achieved so far.
        scheduler: Optional learning rate scheduler; its state is saved if provided.
        train_lora: If True, save only LoRA adapter weights instead of the full model.

    Returns:
        None
    """

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "optimizer_name": optimizer.__class__.__name__,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "all_metrics": all_time_all_metrics
    }

    # Save model weights
    if train_lora:
        checkpoint["model_state_dict"] = lora.lora_state_dict(model)
    else:
        if dist.is_initialized():
            # In DDP, use .module
            checkpoint["model_state_dict"] = model.module.state_dict()
        else:
            checkpoint["model_state_dict"] = (
                model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
            )

    # Save only on rank 0 (or if not using DDP)
    if not dist.is_initialized() or dist.get_rank() == 0:
        torch.save(checkpoint, store_path)


def save_last_weights(
    args: argparse.Namespace,
    model: nn.Module,
    device_ids: List[int],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    all_time_all_metrics,
    best_metric: float,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> None:
    """
    Save the latest training checkpoint for continuation or recovery.

    The checkpoint is always written to:
        {args.results_path}/last_{args.model_type}.ckpt

    This wraps `save_weights` and ensures the latest model/optimizer/scheduler
    states are recorded, along with the current epoch and best metric. In DDP,
    only rank 0 performs the save. Supports both standard and LoRA training.

    Args:
        all_time_all_metrics:
        args: Training arguments. Must define `results_path`, `model_type`,
              and `train_lora`.
        model: Model instance (may be wrapped by DDP/DataParallel).
        device_ids: List of GPU IDs used for training.
        optimizer: Optimizer whose state will be saved.
        epoch: Current training epoch.
        best_metric: Current best validation metric.
        scheduler: Optional learning rate scheduler to save state for.

    Returns:
        None
    """
    store_path = f"{args.results_path}/last_{args.model_type}.ckpt"
    save_weights(
        store_path,
        model,
        device_ids,
        optimizer,
        epoch,
        all_time_all_metrics,
        best_metric,
        scheduler,
        args.train_lora,
    )

def apply_energy_gate(
    waveforms: Dict[str, np.ndarray],
    sample_rate: int = 44100,
    gate_interval: float = 0.1,
    energy_threshold: float = 0.01,
    target_instrument: str = 'vocals'
) -> Dict[str, np.ndarray]:
    """
    对分离结果应用能量门控，将低能量片段置零
    
    Args:
        waveforms: 分离结果字典 {instrument: waveform}
        sample_rate: 采样率
        gate_interval: 检测间隔，秒 (default: 0.1s)
        energy_threshold: 能量阈值，低于此值的片段置零 (default: 0.01)
        target_instrument: 目标乐器，只对该乐器应用门控 (default: 'vocals')
    
    Returns:
        处理后的 waveforms 字典
    """
    if target_instrument not in waveforms:
        return waveforms
    
    audio = waveforms[target_instrument]
    interval_samples = int(gate_interval * sample_rate)
    total_samples = audio.shape[-1]
    
    # 创建副本避免修改原数组
    audio = audio.copy()
    
    gated_count = 0
    total_intervals = 0
    
    for start_idx in range(0, total_samples, interval_samples):
        end_idx = min(start_idx + interval_samples, total_samples)
        segment = audio[..., start_idx:end_idx]
        
        # 计算 RMS 能量
        rms_energy = np.sqrt(np.mean(segment ** 2))
        total_intervals += 1
        
        if rms_energy < energy_threshold:
            audio[..., start_idx:end_idx] = 0.0
            gated_count += 1
    
    waveforms[target_instrument] = audio
    
    return waveforms
