import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import subprocess
import shutil
import tempfile

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix, demix_with_spk
from utils.model_utils import prefer_target_instrument, load_start_checkpoint

from diffusion.diffusion_wrapper import DiffusionWrapper, DiffusionConfig

import warnings

warnings.filterwarnings("ignore")


def load_audio_with_fallback(path: str, sr: int, mono: bool = False):
    """
    Load audio with librosa, fallback to soundfile, then ffmpeg if both fail.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Whether to load as mono
        
    Returns:
        audio: numpy array (channels, samples) or (samples,) if mono
        sample_rate: actual sample rate (should equal sr)
    """
    import scipy.signal
    
    librosa_error = None
    soundfile_error = None
    
    # Try librosa first
    try:
        audio, sample_rate = librosa.load(path, sr=sr, mono=mono)
        return audio, sample_rate
    except Exception as e:
        librosa_error = str(e)
        print(f'[librosa failed] {e}, trying soundfile...')
    
    # Fallback to soundfile
    try:
        audio, orig_sr = sf.read(path, dtype='float32')
        
        # soundfile returns (samples, channels), convert to (channels, samples)
        if len(audio.shape) == 2:
            audio = audio.T  # (samples, channels) -> (channels, samples)
        
        # Convert to mono if requested
        if mono and len(audio.shape) == 2:
            audio = audio.mean(axis=0)
        
        # Resample if needed
        if orig_sr != sr:
            if len(audio.shape) == 1:
                audio = scipy.signal.resample_poly(audio, sr, orig_sr).astype('float32')
            else:
                resampled = []
                for ch in range(audio.shape[0]):
                    resampled.append(
                        scipy.signal.resample_poly(audio[ch], sr, orig_sr).astype('float32')
                    )
                audio = np.stack(resampled, axis=0)
        
        return audio, sr
    except Exception as e:
        soundfile_error = str(e)
        print(f'[soundfile failed] {e}, trying ffmpeg...')
    
    # Fallback to ffmpeg (handles fltp and other formats)
    try:
        audio, sample_rate = load_audio_with_ffmpeg(path, sr, mono)
        return audio, sample_rate
    except Exception as e:
        raise RuntimeError(
            f'All audio loading methods failed for {path}:\n'
            f'  librosa error: {librosa_error}\n'
            f'  soundfile error: {soundfile_error}\n'
            f'  ffmpeg error: {e}'
        )


def load_audio_with_ffmpeg(path: str, sr: int, mono: bool = False):
    """
    Load audio using ffmpeg (handles fltp and other problematic formats).
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Whether to load as mono
        
    Returns:
        audio: numpy array (channels, samples) or (samples,) if mono
        sample_rate: actual sample rate
    """
    import struct
    
    # Build ffmpeg command
    channels = 1 if mono else 2
    cmd = [
        'ffmpeg',
        '-i', path,
        '-f', 'f32le',           # 32-bit float little-endian PCM
        '-acodec', 'pcm_f32le',
        '-ar', str(sr),          # Target sample rate
        '-ac', str(channels),    # Number of channels
        '-v', 'quiet',           # Suppress ffmpeg output
        '-'                      # Output to stdout
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
        
        # Convert raw bytes to numpy array
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        
        if not mono and channels == 2:
            # Reshape to (samples, 2) then transpose to (2, samples)
            audio = audio.reshape(-1, 2).T
        
        return audio, sr
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'ffmpeg failed: {e.stderr.decode() if e.stderr else str(e)}')
    except FileNotFoundError:
        raise RuntimeError('ffmpeg not found. Please install ffmpeg.')


def run_folder(model, args, config, device, verbose: bool = False):
    """
    Process a folder of audio files for source separation.
    """

    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Diffusion model setup
    diffusion_wrappers = {}
    use_diffusion = getattr(args, 'use_diffusion', False)
    
    if use_diffusion:
        diffusion_model_raw = getattr(config.training, 'diffusion_model', None)
        num_stems = len(config.training.instruments)
        diffusion_models = []
        
        if diffusion_model_raw is None:
            print("Warning: diffusion_model not found in config, using default 'dit' for all stems")
            diffusion_models = ['none'] * num_stems
        elif len(diffusion_model_raw) != num_stems:
            raise ValueError(f"diffusion_model length ({len(diffusion_model_raw)}) must match number of instruments ({num_stems})")
        else:
            diffusion_models = diffusion_model_raw
        
        print(f"Loading diffusion models for {num_stems} stems with types: {diffusion_models}")
        
        for idx, (instr, diffusion_type) in enumerate(zip(config.training.instruments, diffusion_models)):
            if diffusion_type is None or diffusion_type.lower() == 'none':
                diffusion_wrappers[instr] = None
                print(f"Stem '{instr}': No diffusion model (None)")
                continue
            
            diff_cfg = DiffusionConfig(
                model_type=diffusion_type,
            )
            
            diff_wrapper = DiffusionWrapper(diff_cfg, device=device)
            
            # Load checkpoint
            diffusion_model_path = getattr(args, 'diffusion_model_path', '')
            if diffusion_model_path != '':
                ckpt_path = os.path.join(diffusion_model_path, f'{instr}_diffusion.pth')
                if os.path.exists(ckpt_path):
                    diff_wrapper.load_checkpoint(ckpt_path)
                    print(f"Loaded diffusion checkpoint for {instr} from {ckpt_path}")
                else:
                    print(f"Warning: diffusion checkpoint not found for {instr} at {ckpt_path}")
            
            diff_wrapper.eval()
            print(f"Stem '{instr}': Diffusion model type '{diffusion_type}'")
            diffusion_wrappers[instr] = diff_wrapper

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    # Define paths for external speaker extraction
    SPEAKER_PYTHON_EXE = "/user-fs/chenzihao/wengzhenqiang/miniconda3/envs/speaker/bin/python"
    SPEAKER_SCRIPT_PATH = "/user-fs/chenzihao/wengzhenqiang/speaker/batch_extract_embeddings.py"
    
    # Directories
    wo_spk_dir = os.path.join(args.store_dir, 'wo_spk')
    with_spk_dir = os.path.join(args.store_dir, 'with_spk')
    embeddings_dir = os.path.join(args.store_dir, 'embeddings')

    # --- PHASE 1: Blind Separation ---
    print("\n" + "="*50)
    print("PHASE 1: Blind Separation (wo_spk)")
    print("="*50)

    for path in tqdm(mixture_paths, desc="Blind Separation"):
        file_name = os.path.splitext(os.path.basename(path))[0]
        output_dir = os.path.join(wo_spk_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if vocals.wav exists to skip (simple resume logic)
        if os.path.exists(os.path.join(output_dir, 'vocals.wav')):
            continue

        try:
            mix, sr = load_audio_with_fallback(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {path}')
            print(f'Error message: {str(e)}')
            continue

        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    mix = np.concatenate([mix, mix], axis=0)

        if 'normalize' in config.inference and config.inference['normalize'] is True:
            mix, norm_params = normalize_audio(mix)
        else:
            norm_params = None

        # Blind inference
        waveforms_blind = demix(config, model, mix, device, model_type=args.model_type, pbar=False)

        # Save results
        for instr, est in waveforms_blind.items():
            if 'normalize' in config.inference and config.inference['normalize'] is True:
                est = denormalize_audio(est, norm_params)
            
            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'
            
            sf.write(os.path.join(output_dir, f"{instr}.{codec}"), est.T, sr, subtype=subtype)
            
            # Ensure vocals.wav exists for embedding extraction
            if instr == 'vocals':
                if codec != 'wav':
                     sf.write(os.path.join(output_dir, 'vocals.wav'), est.T, sr, subtype='FLOAT')

    # --- PHASE 2: Extract Embeddings ---
    print("\n" + "="*50)
    print("PHASE 2: Extracting Embeddings")
    print("="*50)

    if os.path.exists(embeddings_dir):
        shutil.rmtree(embeddings_dir)
    os.makedirs(embeddings_dir, exist_ok=True)

    print(f"Extracting embeddings from {wo_spk_dir}...")
    cmd = [
        SPEAKER_PYTHON_EXE,
        SPEAKER_SCRIPT_PATH,
        "--input_dir", wo_spk_dir,
        "--output_dir", embeddings_dir,
        "--num_samples", "200"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running speaker extraction script: {e}")
        print("Aborting guided separation.")
        return
    except FileNotFoundError:
        print(f"Could not find python executable at {SPEAKER_PYTHON_EXE}")
        return

    # --- PHASE 3: Guided Separation ---
    print("\n" + "="*50)
    print("PHASE 3: Guided Separation (with_spk)")
    print("="*50)

    iterator = tqdm(mixture_paths, desc="Guided Separation") if not verbose else mixture_paths

    for path in iterator:
        if verbose:
            print(f"Processing track: {path}")
        
        file_name = os.path.splitext(os.path.basename(path))[0]
        embedding_path = os.path.join(embeddings_dir, file_name, 'embedding.npy')
        output_dir = os.path.join(with_spk_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            mix, sr = load_audio_with_fallback(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {path}')
            print(f'Error message: {str(e)}')
            continue

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    if verbose: print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)
        else:
            norm_params = None

        # Load embedding
        spk_emb = None
        if os.path.exists(embedding_path):
            try:
                embs = np.load(embedding_path)
                if len(embs) > 0:
                    emb_mean = np.mean(embs, axis=0)
                    spk_emb = torch.from_numpy(emb_mean).to(device).float()
            except Exception as e:
                print(f"Error loading embedding for {file_name}: {e}")
        
        # Inference
        if spk_emb is not None:
            waveforms_orig = demix_with_spk(config, model, mix, spk_emb, device, model_type=args.model_type, pbar=detailed_pbar)
        else:
            if verbose: print("Warning: No embedding found, using blind separation.")
            waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)

        # Apply diffusion model enhancement (if enabled)
        if use_diffusion and diffusion_wrappers:
            if verbose: print("Applying diffusion model enhancement...")
            diffusion_steps = getattr(args, 'diffusion_steps', 10)
            
            for instr in instruments:
                if instr in diffusion_wrappers and diffusion_wrappers[instr] is not None:
                    diff_model = diffusion_wrappers[instr]
                    
                    # waveforms_orig[instr]: [C, T] numpy array
                    # Convert to torch tensor: [C, T] -> [1, C, T]
                    source_wave = torch.from_numpy(waveforms_orig[instr]).unsqueeze(0).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        enhanced_wave = diff_model.inference(
                            source_wave,  # [1, C, T]
                            num_steps=diffusion_steps,
                            method='euler'
                        )
                    
                    # Convert back to numpy: [1, C, T] -> [C, T]
                    waveforms_orig[instr] = enhanced_wave.squeeze(0).cpu().numpy()
                    if verbose: print(f"Enhanced {instr} with diffusion model")

        # Post-processing
        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')
            
        if args.extract_other:
            for instr in config.training.instruments:
                if instr != "other":
                    waveforms_orig['other'] = waveforms_orig.get('other', mix_orig) - waveforms_orig[instr]
            if 'other' not in instruments:
                instruments.append('other')

        # Save results
        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = os.path.join(output_dir, f"{instr}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{instr}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
