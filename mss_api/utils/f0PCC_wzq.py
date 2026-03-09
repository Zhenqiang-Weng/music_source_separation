# -*- coding: utf-8 -*-
"""
使用RMVPE提取F0后计算logF0PCC的脚本

功能：
1. 使用RMVPE模型提取音频的F0
2. 计算两个音频之间的logF0 Pearson相关系数(PCC)
3. 支持批量处理音频文件对
4. 输出详细的统计信息

依赖：
    pip install librosa numpy scipy torch soundfile
    需要RMVPE模型权重文件
"""

import os
import sys
import argparse
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.stats import pearsonr
import soundfile as sf

# 添加项目根目录到路径
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from modules.rmvpe import RMVPE


def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，用于JSON序列化
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


class F0PCCCalculator:
    """F0 PCC计算器"""
    
    def __init__(self, model_path: str, device: str = None, is_half: bool = False):
        """
        初始化F0PCC计算器
        
        Args:
            model_path: RMVPE模型路径
            device: 计算设备 ('cuda', 'cpu', 'mps')
            is_half: 是否使用半精度
        """
        self.device = device or self._get_device()
        self.is_half = is_half
        
        # 初始化RMVPE模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RMVPE模型文件不存在: {model_path}")
        
        print(f"正在加载RMVPE模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        self.rmvpe = RMVPE(
            model_path=model_path,
            is_half=is_half,
            device=self.device
        )
        print("RMVPE模型加载完成")
    
    def _get_device(self) -> str:
        """自动选择计算设备"""
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def extract_f0(self, audio_path: str, target_sr: int = 16000, thred: float = 0.03) -> Optional[np.ndarray]:
        """
        使用RMVPE提取音频的F0
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            thred: F0提取阈值
            
        Returns:
            F0序列 (numpy数组) 或 None (如果提取失败)
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # 提取F0
            f0 = self.rmvpe.infer_from_audio(audio, thred=thred)
            return f0
            
        except Exception as e:
            print(f"[错误] 提取F0失败 {audio_path}: {e}")
            return None
    
    def calculate_logf0_pcc(self, f0_1: np.ndarray, f0_2: np.ndarray) -> Dict:
        """
        计算两个F0序列的logF0 PCC
        
        Args:
            f0_1: 第一个F0序列
            f0_2: 第二个F0序列
            
        Returns:
            包含PCC结果的字典
        """
        # 确保两个序列长度一致
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]
        
        # 找到两个序列中都有声音的帧
        voiced_mask_1 = f0_1 > 0
        voiced_mask_2 = f0_2 > 0
        common_voiced_mask = voiced_mask_1 & voiced_mask_2
        
        if np.sum(common_voiced_mask) < 10:  # 至少需要10个有声帧
            return {
                'pcc': 0.0,
                'p_value': 1.0,
                'voiced_frames_1': np.sum(voiced_mask_1),
                'voiced_frames_2': np.sum(voiced_mask_2),
                'common_voiced_frames': np.sum(common_voiced_mask),
                'total_frames': min_len,
                'error': 'insufficient_voiced_frames'
            }
        
        # 提取共同有声帧的F0值
        voiced_f0_1 = f0_1[common_voiced_mask]
        voiced_f0_2 = f0_2[common_voiced_mask]
        
        # 转换为log域
        log_f0_1 = np.log(voiced_f0_1 + 1e-8)  # 添加小值避免log(0)
        log_f0_2 = np.log(voiced_f0_2 + 1e-8)
        
        # 计算Pearson相关系数
        try:
            pcc, p_value = pearsonr(log_f0_1, log_f0_2)
            
            return {
                'pcc': float(pcc),
                'p_value': float(p_value),
                'voiced_frames_1': int(np.sum(voiced_mask_1)),
                'voiced_frames_2': int(np.sum(voiced_mask_2)),
                'common_voiced_frames': int(np.sum(common_voiced_mask)),
                'total_frames': int(min_len),
                'mean_log_f0_1': float(np.mean(log_f0_1)),
                'mean_log_f0_2': float(np.mean(log_f0_2)),
                'std_log_f0_1': float(np.std(log_f0_1)),
                'std_log_f0_2': float(np.std(log_f0_2))
            }
            
        except Exception as e:
            return {
                'pcc': 0.0,
                'p_value': 1.0,
                'voiced_frames_1': int(np.sum(voiced_mask_1)),
                'voiced_frames_2': int(np.sum(voiced_mask_2)),
                'common_voiced_frames': int(np.sum(common_voiced_mask)),
                'total_frames': int(min_len),
                'error': str(e)
            }
    
    def calculate_pcc_for_pair(self, audio_path_1: str, audio_path_2: str) -> Dict:
        """
        计算一对音频的logF0 PCC
        
        Args:
            audio_path_1: 第一个音频文件路径
            audio_path_2: 第二个音频文件路径
            
        Returns:
            包含完整结果的字典
        """
        result = {
            'audio_1': audio_path_1,
            'audio_2': audio_path_2,
            'success': False
        }
        
        # 提取F0
        print(f"正在处理: {os.path.basename(audio_path_1)} vs {os.path.basename(audio_path_2)}")
        
        f0_1 = self.extract_f0(audio_path_1)
        if f0_1 is None:
            result['error'] = f"无法提取F0: {audio_path_1}"
            return result
        
        f0_2 = self.extract_f0(audio_path_2)
        if f0_2 is None:
            result['error'] = f"无法提取F0: {audio_path_2}"
            return result
        
        # 计算PCC
        pcc_result = self.calculate_logf0_pcc(f0_1, f0_2)
        result.update(pcc_result)
        result['success'] = True
        
        print(f"  logF0 PCC: {pcc_result['pcc']:.4f} (p={pcc_result['p_value']:.4f})")
        print(f"  共同有声帧: {pcc_result['common_voiced_frames']}/{pcc_result['total_frames']}")
        
        return result
    
    def batch_calculate_pcc(self, audio_pairs: List[Tuple[str, str]]) -> Dict:
        """
        批量计算多对音频的logF0 PCC
        
        Args:
            audio_pairs: 音频文件对列表
            
        Returns:
            包含所有结果的字典
        """
        results = []
        successful_pccs = []
        
        print(f"开始批量处理 {len(audio_pairs)} 对音频文件...")
        
        for i, (audio_1, audio_2) in enumerate(audio_pairs, 1):
            print(f"\n[{i}/{len(audio_pairs)}]", end=" ")
            
            result = self.calculate_pcc_for_pair(audio_1, audio_2)
            results.append(result)
            
            if result['success'] and 'error' not in result:
                successful_pccs.append(result['pcc'])
        
        # 计算统计信息
        stats = self._calculate_statistics(successful_pccs)
        
        return {
            'results': results,
            'statistics': stats,
            'total_pairs': len(audio_pairs),
            'successful_pairs': len(successful_pccs)
        }
    
    def _calculate_statistics(self, pccs: List[float]) -> Dict:
        """计算PCC统计信息"""
        if not pccs:
            return {'count': 0, 'error': 'no_valid_results'}
        
        pccs = np.array(pccs)
        
        return {
            'count': len(pccs),
            'mean': float(np.mean(pccs)),
            'std': float(np.std(pccs)),
            'min': float(np.min(pccs)),
            'max': float(np.max(pccs)),
            'median': float(np.median(pccs)),
            'q25': float(np.percentile(pccs, 25)),
            'q75': float(np.percentile(pccs, 75))
        }


def load_audio_pairs_from_file(file_path: str) -> List[Tuple[str, str]]:
    """
    从文件加载音频对列表
    
    文件格式：每行两个音频路径，用制表符分隔
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                print(f"[警告] 第{line_num}行格式错误，跳过: {line}")
                continue
            
            audio_1, audio_2 = parts
            if not os.path.exists(audio_1):
                print(f"[警告] 文件不存在: {audio_1}")
                continue
            if not os.path.exists(audio_2):
                print(f"[警告] 文件不存在: {audio_2}")
                continue
            
            pairs.append((audio_1, audio_2))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="使用RMVPE提取F0并计算logF0 PCC")
    parser.add_argument("--model_path", type=str, default='./checkpoints/rmvpe/rmvpe.pt',
                        help="RMVPE模型路径")
    parser.add_argument("--audio1", type=str, help="第一个音频文件路径")
    parser.add_argument("--audio2", type=str, help="第二个音频文件路径")
    parser.add_argument("--pairs_file", type=str, 
                        help="音频对列表文件路径（每行两个音频路径）")
    parser.add_argument("--output", type=str, default="f0pcc_results.json",
                        help="输出结果文件路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], 
                        help="计算设备")
    parser.add_argument("--half", action="store_true", help="使用半精度")
    parser.add_argument("--target_sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--threshold", type=float, default=0.03, help="F0提取阈值")

    args = parser.parse_args()
    args.audio1 = '/user-fs/chenzihao/chengongyu/svc/seed-vc/outputs/8.18-famous-accom-8.14v2model/vc_apt_(Vocals)_川普_1.0_100_0.7.wav'
    args.audio2 = '/user-fs/chenzihao/chengongyu/svc/seed-vc/outputs/8.18-famous-accom-8.14v2model/vc_apt_(Vocals)_贾斯汀比伯_1.0_100_0.7.wav'
    # 检查参数
    if not args.audio1 and not args.pairs_file:
        parser.error("必须指定 --audio1 和 --audio2，或者 --pairs_file")
    
    if args.audio1 and not args.audio2:
        parser.error("指定 --audio1 时必须同时指定 --audio2")
    
    # 初始化计算器
    calculator = F0PCCCalculator(
        model_path=args.model_path,
        device=args.device,
        is_half=args.half
    )
    
    # 准备音频对
    if args.pairs_file:
        print(f"从文件加载音频对: {args.pairs_file}")
        audio_pairs = load_audio_pairs_from_file(args.pairs_file)
        print(f"加载了 {len(audio_pairs)} 对音频")
    else:
        audio_pairs = [(args.audio1, args.audio2)]
    
    if not audio_pairs:
        print("没有有效的音频对，退出")
        return
    
    # 计算PCC
    results = calculator.batch_calculate_pcc(audio_pairs)
    
    # 保存结果（转换numpy类型）
    print(f"\n正在保存结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(results), f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("统计结果:")
    print(f"总音频对数: {results['total_pairs']}")
    print(f"成功处理数: {results['successful_pairs']}")
    
    if results['statistics']['count'] > 0:
        stats = results['statistics']
        print(f"logF0 PCC统计:")
        print(f"  平均值: {stats['mean']:.4f}")
        print(f"  标准差: {stats['std']:.4f}")
        print(f"  最小值: {stats['min']:.4f}")
        print(f"  最大值: {stats['max']:.4f}")
        print(f"  中位数: {stats['median']:.4f}")
        print(f"  25%分位数: {stats['q25']:.4f}")
        print(f"  75%分位数: {stats['q75']:.4f}")
    
    print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
