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
import argparse
import math
import torch.multiprocessing as mp  # 引入多进程库
from types import SimpleNamespace

try:
    import scipy.signal
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config
from utils.model_utils import demix, demix_with_spk_2, load_start_checkpoint
from spk_extract import load_model as load_spk_model

import warnings
warnings.filterwarnings("ignore")


def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    if _HAS_SCIPY:
        C, _ = audio.shape
        out = []
        for c in range(C):
            out.append(scipy.signal.resample_poly(audio[c], target_sr, orig_sr).astype(np.float32))
        return np.stack(out, axis=0)
    
    C, N = audio.shape
    duration = N / float(orig_sr)
    new_n = int(round(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=N, endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_n, endpoint=False)
    out = []
    for c in range(C):
        out.append(np.interp(x_new, x_old, audio[c]).astype(np.float32))
    return np.stack(out, axis=0)


def _ensure_2ch(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)
    return audio


class AudioSeparator:
    def __init__(self,
                 config_path,
                 model_path,
                 model_type='bdc_sg_bs_roformer',
                 device_ids=[0],
                 force_cpu=False,
                 use_autocast=True,
                 autocast_dtype="fp16",
                 **kwargs):
        """
        初始化分离器
        """
        self.args = SimpleNamespace()
        self.args.config_path = config_path
        self.args.start_check_point = model_path
        self.args.model_type = model_type
        self.args.device_ids = device_ids
        self.args.force_cpu = force_cpu

        # 默认参数配置
        defaults = {
            'spk_model_path': 'ckpt/speech_campplus_sv_zh_en_16k-common_advanced',
            'segment_duration': 2,
            'energy_threshold': 0.1,
            'max_clusters': 2,
            'apply_gate': False,
            'gate_interval': 0.1,
            'gate_threshold': 0.01,
            'flac_file': False,
            'pcm_type': 'PCM_16',
            'lora_checkpoint': None,
            'output_name_mapping': {
                'vocals': 'leading_vocal',
            }
        }

        for k, v in defaults.items():
            setattr(self.args, k, kwargs.get(k, v))

        self.use_autocast = bool(use_autocast)
        self.autocast_dtype = autocast_dtype.lower().strip()

        # 设备设置
        if self.args.force_cpu:
            self.device = "cpu"
            print("[Init] Using CPU (--force_cpu)")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            
            # 注意：如果是多进程模式调用，这里的 device_ids 应该只包含一个 ID
            if isinstance(self.args.device_ids, list) and len(self.args.device_ids) > 0:
                self.device = f'cuda:{self.args.device_ids[0]}'
            else:
                self.device = 'cuda:0'
            
            torch.backends.cudnn.benchmark = True
            # 只有在非多进程分发模式下才打印初始化信息，避免刷屏
            if kwargs.get('verbose', True):
                print(f"[Init] Using CUDA device: {self.device}")

        # 加载分离模型
        if kwargs.get('verbose', True):
            print(f"[Init] Loading Separation Model ({self.args.model_type})...")
        
        self.model, self.config = get_model_from_config(self.args.model_type, self.args.config_path)

        if self.args.start_check_point:
            if kwargs.get('verbose', True):
                print(f"[Init] Loading weights from: {self.args.start_check_point}")
            checkpoint = torch.load(self.args.start_check_point, map_location='cpu', weights_only=False)
            load_start_checkpoint(self.args, self.model, checkpoint, type_='inference')


        if isinstance(self.args.device_ids, list) and len(self.args.device_ids) > 1 and not self.args.force_cpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载说话人模型
        if kwargs.get('verbose', True):
            print(f"[Init] Loading Speaker Model...")
        try:
            self.spk_model = load_spk_model(self.args.spk_model_path, device=str(self.device))
            self.spk_model.eval()
        except Exception as e:
            print(f"[Error] Failed to load speaker model: {e}")
            self.spk_model = None

        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100)

    def _load_audio(self, path: str):
        if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
        if os.path.getsize(path) < 100: raise RuntimeError(f"File too small: {path}")
        sr = self.sample_rate
        lower_path = path.lower()
        is_compressed = lower_path.endswith(('.mp3', '.m4a', '.aac', '.ogg', '.wma'))
        
        if is_compressed:
            try: return self._load_audio_ffmpeg(path, sr)
            except: pass
        else:
            try:
                audio, orig_sr = sf.read(path, dtype='float32', always_2d=True)
                audio = audio.T
                audio = _ensure_2ch(audio)
                if orig_sr != sr: audio = _resample_np(audio, orig_sr, sr)
                return audio, sr
            except:
                try: return self._load_audio_ffmpeg(path, sr)
                except: pass
        try:
            audio, _ = librosa.load(path, sr=sr, mono=False)
            if audio.ndim == 1: audio = audio[None, :]
            audio = _ensure_2ch(audio.astype(np.float32))
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"All load methods failed for {path}. Last error: {e}")

    def _load_audio_ffmpeg(self, path: str, sr: int):
        cmd = ['ffmpeg', '-err_detect', 'ignore_err', '-ignore_unknown', '-y', '-i', path, '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sr), '-ac', '2', '-vn', '-sn', '-map_metadata', '-1', '-v', 'quiet', '-']
        try:
            res = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            audio = np.frombuffer(res.stdout, dtype=np.float32)
            audio = audio.reshape(-1, 2).T
            return audio, sr
        except subprocess.TimeoutExpired: raise RuntimeError("FFmpeg timeout")
        except Exception as e: raise RuntimeError(f"FFmpeg failed: {e}")

    def _maybe_autocast_ctx(self):
        if self.args.force_cpu or (not self.use_autocast):
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False)
        if self.autocast_dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)

    def separate_audio(self, audio_path: str):
        try:
            mix, _ = self._load_audio(audio_path)
        except Exception as e:
            print(f"[Skip] Load failed {audio_path}: {e}")
            return {}

        mix = _ensure_2ch(mix)
        mix_orig = mix.copy()

        norm_params = None
        if getattr(self.config, 'inference', None) is not None and self.config.inference.get('normalize', False):
            mix, norm_params = normalize_audio(mix)

        with torch.inference_mode():
            with self._maybe_autocast_ctx():
                # [新增] 打印模型结构和参数精度
                # self._print_model_info_once()
                
                # [新增] 验证 autocast 是否生效（检查前向传播时的 tensor 精度）
                # self._check_autocast_active()
                
                waveforms = demix_with_spk_2(
                    config=self.config, model=self.model, mix=mix, spk_model=self.spk_model,
                    device=self.device, 
                    segment_duration=self.args.segment_duration, energy_threshold=self.args.energy_threshold,
                    max_clusters=self.args.max_clusters,
                    apply_gate=self.args.apply_gate, gate_interval=self.args.gate_interval,
                    gate_threshold=self.args.gate_threshold
                )

        if getattr(self.config, 'inference', None) is not None and self.config.inference.get('normalize', False) and norm_params is not None:
            for key in list(waveforms.keys()):
                waveforms[key] = denormalize_audio(waveforms[key], norm_params)
        
        return waveforms

    def demix_once(self, audio_path: str):
        """
        Load audio and run a single blind demix pass (demix) and return the results.
        This skips speaker-guided refinement (demix_with_spk_2).
        """
        try:
            mix, _ = self._load_audio(audio_path)
        except Exception as e:
            print(f"[Skip] Load failed {audio_path}: {e}")
            return {}

        mix = _ensure_2ch(mix)
        norm_params = None
        if getattr(self.config, 'inference', None) is not None and self.config.inference.get('normalize', False):
            mix, norm_params = normalize_audio(mix)

        try:
            # demix expects a tensor/ndarray and will run inference once
            waveforms = demix(self.config, self.model, mix, device=self.device)
        except Exception as e:
            print(f"[Error] demix failed for {audio_path}: {e}")
            return {}

        if getattr(self.config, 'inference', None) is not None and self.config.inference.get('normalize', False) and norm_params is not None:
            for key in list(waveforms.keys()):
                waveforms[key] = denormalize_audio(waveforms[key], norm_params)

        return waveforms

    def _print_model_info_once(self):
        """打印模型结构和参数精度（只执行一次）"""
        if hasattr(self, '_model_info_printed'):
            return
        self._model_info_printed = True
        
        output_file = "model_structure_and_precision.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Model Structure and Parameter Precision\n")
                f.write("=" * 80 + "\n\n")
                
                # 获取实际的模型（处理 DataParallel）
                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                
                # 统计信息
                total_params = 0
                dtype_counts = {}
                
                f.write("Layer-wise Parameter Information:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Layer Name':<60} {'Shape':<25} {'Dtype':<10} {'Device':<10}\n")
                f.write("-" * 80 + "\n")
                
                for name, param in model.named_parameters():
                    shape_str = str(tuple(param.shape))
                    dtype_str = str(param.dtype).replace('torch.', '')
                    device_str = str(param.device)
                    
                    f.write(f"{name:<60} {shape_str:<25} {dtype_str:<10} {device_str:<10}\n")
                    
                    total_params += param.numel()
                    dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + param.numel()
                
                # 统计摘要
                f.write("\n" + "=" * 80 + "\n")
                f.write("Summary:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Total Parameters (M): {total_params / 1e6:.2f}M\n\n")
                
                f.write("Parameter Dtype Distribution:\n")
                for dtype, count in sorted(dtype_counts.items()):
                    percentage = (count / total_params) * 100
                    f.write(f"  {dtype:<15} {count:>12,} ({percentage:>6.2f}%)\n")
                
                # 模型结构
                f.write("\n" + "=" * 80 + "\n")
                f.write("Model Architecture:\n")
                f.write("-" * 80 + "\n")
                f.write(str(model))
                
            print(f"\n[Info] Model structure saved to: {output_file}")
            print(f"[Info] Total parameters: {total_params / 1e6:.2f}M")
            for dtype, count in sorted(dtype_counts.items()):
                percentage = (count / total_params) * 100
                print(f"[Info]   {dtype}: {percentage:.2f}%")
                
        except Exception as e:
            print(f"[Warning] Failed to save model info: {e}")

    def _check_autocast_active(self):
        """检查 autocast 是否在前向传播时生效"""
        if hasattr(self, '_autocast_checked'):
            return
        self._autocast_checked = True
        
        try:
            # 创建一个小的测试 tensor 做矩阵乘法
            test_a = torch.randn(16, 16, device=self.device, dtype=torch.float32)
            test_b = torch.randn(16, 16, device=self.device, dtype=torch.float32)
            
            # 在 autocast 上下文中做运算
            result = torch.matmul(test_a, test_b)
            
            # 检查结果的 dtype
            print(f"\n[Autocast Check]")
            print(f"  Input dtype: {test_a.dtype}")
            print(f"  Output dtype: {result.dtype}")
            print(f"  Autocast enabled: {self.use_autocast}")
            print(f"  Autocast dtype: {self.autocast_dtype}")
            
            if self.use_autocast:
                if self.autocast_dtype == "bf16" and result.dtype == torch.bfloat16:
                    print(f"  ✓ Autocast is working (BF16)")
                elif self.autocast_dtype == "fp16" and result.dtype == torch.float16:
                    print(f"  ✓ Autocast is working (FP16)")
                else:
                    print(f"  ⚠️ Autocast may not be working correctly")
            else:
                print(f"  Autocast disabled, using FP32")
                
        except Exception as e:
            print(f"[Warning] Autocast check failed: {e}")

    def inference_file(self, audio_path: str, output_root_dir: str):
            t0 = time.time()

            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(output_root_dir, file_name)

            waveforms = self.separate_audio(audio_path)
            if not waveforms: return None
            
            os.makedirs(output_dir, exist_ok=True)

            codec = 'flac' if self.args.flac_file else 'wav'
            subtype = 'PCM_16' if self.args.pcm_type == 'PCM_16' else 'FLOAT'
            sr = self.sample_rate

            name_mapping = getattr(self.args, 'output_name_mapping', {})

            for instr, estimates in waveforms.items():
                save_name = name_mapping.get(instr, instr)
                out_path = os.path.join(output_dir, f"{save_name}.{codec}")
                sf.write(out_path, estimates.T, sr, subtype=subtype)
            
            # [新增] 2. 计算耗时并打印日志
            elapsed = time.time() - t0
            print(f"[{file_name}] Processed in {elapsed:.2f}s")
            
            return output_dir

    def inference_folder(self, input_folder: str, output_root_dir: str):
        extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a', '*.aac', '*.ogg']
        files = []
        for ext in extensions: files.extend(glob.glob(os.path.join(input_folder, ext)))
        files = sorted(set([p for p in files if os.path.isfile(p)]))
        print(f"[Batch] Found {len(files)} files in {input_folder}")
        for path in tqdm(files, desc="Batch Inference"):
            try: self.inference_file(path, output_root_dir)
            except Exception as e: print(f"[Batch] Error {path}: {e}")

    def inference_from_list(self, txt_path: str, output_root_dir: str):
        if not os.path.exists(txt_path):
            print(f"[Error] List not found: {txt_path}")
            return
        with open(txt_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        files = [line.strip().strip('"').strip("'") for line in lines if line.strip()]
        files = sorted(set(files))
        
        print(f"[List] Processing {len(files)} files...")
        t0 = time.time()
        for path in tqdm(files, desc="List Inference"):
            if not os.path.exists(path):
                print(f"[Warning] Missing: {path}")
                continue
            try: self.inference_file(path, output_root_dir)
            except Exception as e: print(f"[Error] {path}: {e}")
        print(f"[List] Done. Elapsed {time.time() - t0:.2f}s")


# =================================================================
# [新增] 多卡并行处理逻辑
# =================================================================
def _worker_process(gpu_id, file_list, args_dict, output_dir):
    """
    工作进程：初始化模型并处理分配到的文件列表
    """
    # 每个进程只看到自己的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device_id = 0 # 相对 ID 变为 0
    
    # 重新构建 AudioSeparator (在子进程中)
    # 注意：args_dict 是从主进程传过来的字典配置
    try:
        separator = AudioSeparator(
            config_path=args_dict['config_path'],
            model_path=args_dict['start_check_point'],
            model_type=args_dict['model_type'],
            device_ids=[device_id], # 只传一个ID
            force_cpu=False,
            use_autocast=(not args_dict['no_autocast']),
            autocast_dtype=args_dict['autocast_dtype'],
            spk_model_path=args_dict['spk_model_path'],
            flac_file=(args_dict['format'] == 'flac'),
            verbose=False # 减少子进程日志
        )
        
        # tqdm 的 position 参数可以让多个进度条共存，但这里简单起见，只打印日志或用总进度条
        # 建议：子进程简单打印即可
        print(f"[GPU {gpu_id}] Start processing {len(file_list)} files...")
        
        for path in file_list:
            if not os.path.exists(path): continue
            try:
                separator.inference_file(path, output_dir)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error {os.path.basename(path)}: {e}")
                
        print(f"[GPU {gpu_id}] Finished.")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Crashed: {e}")
        import traceback
        traceback.print_exc()


def run_distributed_inference(txt_path, output_dir, device_ids, args_namespace):
    """
    主控函数：切分任务并分发给多张卡
    """
    if not os.path.exists(txt_path):
        print(f"List file not found: {txt_path}")
        return

    # 1. 读取所有文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    files = [line.strip().strip('"').strip("'") for line in lines if line.strip()]
    files = sorted(set(files))
    
    total_files = len(files)
    num_gpus = len(device_ids)
    
    if total_files == 0:
        print("No files to process.")
        return

    print(f"--- Distributed Inference ---")
    print(f"Total files: {total_files}")
    print(f"Available GPUs: {device_ids}")
    
    # 2. 平均切分列表
    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # 3. 准备参数字典 (Namespace 不好跨进程传，转成 dict)
    args_dict = vars(args_namespace)
    
    # 4. 启动进程
    processes = []
    mp.set_start_method('spawn', force=True) # CUDA 必须用 spawn
    
    for i, gpu_id in enumerate(device_ids):
        if i >= len(chunks): break # 防止 GPU 多于任务块
        
        file_chunk = chunks[i]
        p = mp.Process(
            target=_worker_process,
            args=(gpu_id, file_chunk, args_dict, output_dir)
        )
        p.start()
        processes.append(p)
        print(f"Launched worker on GPU {gpu_id} with {len(file_chunk)} files.")
    
    # 5. 等待结束
    for p in processes:
        p.join()
    
    print("All distributed tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bdc_sg_bs_roformer')
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--start_check_point", type=str, required=True)
    parser.add_argument("--input_folder", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--input_list", type=str, default=None)
    parser.add_argument("--store_dir", type=str, required=True)
    # 允许传入多个ID，例如 --device_ids 0 1 2 3
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0])
    parser.add_argument("--spk_model_path", type=str, default="ckpt/speech_campplus_sv_zh_en_16k-common_advanced")
    parser.add_argument("--no_autocast", action="store_true")
    parser.add_argument("--autocast_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "flac"])

    args, unknown = parser.parse_known_args()
    is_flac = (args.format == 'flac')

    if args.input_list and len(args.device_ids) > 1 and not args.force_cpu:
        # 进入分布式逻辑
        run_distributed_inference(args.input_list, args.store_dir, args.device_ids, args)
    else:
        # 传统的单进程逻辑 (初始化一次)
        separator = AudioSeparator(
            config_path=args.config_path,
            model_path=args.start_check_point,
            model_type=args.model_type,
            device_ids=args.device_ids,
            spk_model_path=args.spk_model_path,
            force_cpu=args.force_cpu,
            use_autocast=(not args.no_autocast),
            autocast_dtype=args.autocast_dtype,
            flac_file=is_flac,
            # output_name_mapping={'vocals': 'leading_vocal'}
        )

        if args.input_list: separator.inference_from_list(args.input_list, args.store_dir)
        elif args.input_folder: separator.inference_folder(args.input_folder, args.store_dir)
        elif args.input_file: separator.inference_file(args.input_file, args.store_dir)
        else: print("Please provide --input_folder, --input_file, or --input_list")