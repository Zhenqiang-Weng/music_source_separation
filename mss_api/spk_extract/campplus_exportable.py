"""
CAMP++ TensorRT Export and Inference
Optimized for fixed input shapes: 2s audio at 16kHz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pathlib import Path
import time

# ============== 固定参数 ==============
FEAT_DIM = 80               # Fbank 特征维度
EMBEDDING_SIZE = 192        # 输出 embedding 维度
SAMPLE_RATE = 16000         # 目标采样率
SEGMENT_DURATION = 2.0      # 2秒片段

# Fbank 参数 (与 torchaudio.compliance.kaldi.fbank 一致)
FRAME_LENGTH_MS = 25.0      # 帧长 25ms
FRAME_SHIFT_MS = 10.0       # 帧移 10ms

# 计算固定的 time_frames
# 2秒 = 2000ms, 帧移 10ms -> (2000 - 25) / 10 + 1 = 198.5 -> 198 帧
# 或者用采样点计算: 2s * 16000 = 32000 samples
# frame_length = 25ms * 16 = 400 samples
# frame_shift = 10ms * 16 = 160 samples  
# num_frames = (32000 - 400) / 160 + 1 = 198.75 -> 198 帧
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)  # 32000
FRAME_LENGTH_SAMPLES = int(FRAME_LENGTH_MS * SAMPLE_RATE / 1000)  # 400
FRAME_SHIFT_SAMPLES = int(FRAME_SHIFT_MS * SAMPLE_RATE / 1000)    # 160
TIME_FRAMES = (SEGMENT_SAMPLES - FRAME_LENGTH_SAMPLES) // FRAME_SHIFT_SAMPLES + 1  # 198


class CAMPPlusExportable(nn.Module):
    """
    可导出的 CAMP++ 模型包装器
    固定输入形状: (batch, TIME_FRAMES, FEAT_DIM) = (batch, 198, 80)
    输出: (batch, EMBEDDING_SIZE) = (batch, 192)
    """
    
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.model = original_model
        self.model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 198, 80) - 2秒音频的 Fbank 特征
        Returns:
            embedding: (batch, 192)
        """
        return self.model(x)


def export_campplus_to_onnx(
    model_dir: str,
    output_path: str,
    batch_size: int = 1,
    time_frames: int = TIME_FRAMES,
    feat_dim: int = FEAT_DIM
) -> str:
    """
    导出 CAMP++ 到 ONNX
    
    固定输入: (batch, 198, 80) - 2秒 16kHz 音频的 Fbank 特征
    """
    from . import load_model
    
    print(f"Loading CAMP++ model from: {model_dir}")
    model = load_model(model_dir, device='cpu')
    model.eval()
    
    # 包装模型
    exportable = CAMPPlusExportable(model)
    
    # 创建 dummy 输入 - 固定形状
    dummy_input = torch.randn(batch_size, time_frames, feat_dim)
    
    print(f"\n=== CAMP++ Export Config ===")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Segment duration: {SEGMENT_DURATION}s")
    print(f"  Segment samples: {SEGMENT_SAMPLES}")
    print(f"  Frame length: {FRAME_LENGTH_MS}ms ({FRAME_LENGTH_SAMPLES} samples)")
    print(f"  Frame shift: {FRAME_SHIFT_MS}ms ({FRAME_SHIFT_SAMPLES} samples)")
    print(f"  Time frames: {time_frames}")
    print(f"  Feat dim: {feat_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {tuple(dummy_input.shape)}")
    
    # 测试前向传播
    with torch.no_grad():
        output = exportable(dummy_input)
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Output - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    
    # 导出 ONNX
    print(f"\nExporting to: {output_path}")
    torch.onnx.export(
        exportable,
        dummy_input,
        output_path,
        input_names=['fbank_features'],
        output_names=['embedding'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None  # 固定形状
    )
    
    print(f"✓ ONNX exported: {output_path}")
    return output_path


def onnx_to_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    workspace_gb: int = 4
) -> Optional[str]:
    """将 ONNX 转换为 TensorRT 引擎"""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed")
        return None
    
    print(f"\nConverting to TensorRT...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {output_path}")
    print(f"  FP16: {fp16}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            return None
    
    print("✓ ONNX parsed")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 enabled")
    
    print("Building TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("Build failed")
        return None
    
    with open(output_path, 'wb') as f:
        f.write(bytes(engine_bytes))
    
    engine_size = engine_bytes.nbytes if hasattr(engine_bytes, 'nbytes') else len(bytes(engine_bytes))
    print(f"✓ TensorRT saved: {output_path}")
    print(f"  Engine size: {engine_size / 1024 / 1024:.1f} MB")
    
    return output_path


class CAMPPlusTRT:
    """CAMP++ TensorRT 推理"""
    
    def __init__(self, engine_path: str, device_id: int = 0, use_cuda_graph: bool = True):
        import tensorrt as trt
        
        self.device_id = device_id
        self.use_cuda_graph = use_cuda_graph
        self.device = f'cuda:{device_id}'
        torch.cuda.set_device(device_id)
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"Loading CAMP++ TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 解析 IO
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_name = name
                print(f"  Input: {name} {shape}")
            else:
                self.output_shape = shape
                self.output_name = name
                print(f"  Output: {name} {shape}")
        
        self.batch_size = self.input_shape[0]
        self.time_frames = self.input_shape[1]
        self.feat_dim = self.input_shape[2]
        self.embedding_size = self.output_shape[1]
        
        self.stream = torch.cuda.Stream(device=device_id)
        
        # 预分配 buffers
        self.input_buffer = torch.zeros(self.input_shape, dtype=torch.float32, device=self.device)
        self.output_buffer = torch.zeros(self.output_shape, dtype=torch.float32, device=self.device)
        
        # 设置地址
        self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())
        
        # 构建 CUDA Graph
        self.graph = None
        if use_cuda_graph:
            self._build_cuda_graph()
        
        print(f"✓ CAMP++ TRT initialized")
        print(f"  Input: ({self.batch_size}, {self.time_frames}, {self.feat_dim})")
        print(f"  Output: ({self.batch_size}, {self.embedding_size})")
    
    def _build_cuda_graph(self):
        print("Building CUDA Graph...")
        
        for _ in range(5):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        
        print("✓ CUDA Graph built")
    
    def infer(self, features: torch.Tensor) -> torch.Tensor:
        """
        执行推理
        features: (batch, 198, 80)
        返回: (batch, 192)
        """
        actual_batch = features.shape[0]
        
        self.input_buffer[:actual_batch].copy_(features)
        
        if self.use_cuda_graph and self.graph is not None:
            self.graph.replay()
        else:
            self.context.execute_async_v3(self.stream.cuda_stream)
        
        self.stream.synchronize()
        
        return self.output_buffer[:actual_batch].clone()
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.infer(features)


class SpeakerExtractorTRT:
    """完整的说话人特征提取器 (使用 TensorRT)"""
    
    def __init__(self, engine_path: str, device_id: int = 0, use_cuda_graph: bool = True):
        import torchaudio.compliance.kaldi as Kaldi
        self.Kaldi = Kaldi
        
        self.device_id = device_id
        self.device = f'cuda:{device_id}'
        torch.cuda.set_device(device_id)
        
        self.model = CAMPPlusTRT(engine_path, device_id, use_cuda_graph)
        
        self.sample_rate = SAMPLE_RATE
        self.segment_samples = SEGMENT_SAMPLES
        self.expected_frames = self.model.time_frames
    
    def extract_fbank(self, audio: torch.Tensor) -> torch.Tensor:
        """
        从音频提取 Fbank 特征
        audio: (samples,) 或 (batch, samples) - 必须是 16kHz
        返回: (batch, 198, 80)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        features_list = []
        
        for i in range(batch_size):
            feat = self.Kaldi.fbank(
                audio[i:i+1],
                num_mel_bins=FEAT_DIM,
                sample_frequency=self.sample_rate,
                frame_length=FRAME_LENGTH_MS,
                frame_shift=FRAME_SHIFT_MS
            )
            # CMVN
            feat = feat - feat.mean(dim=0, keepdim=True)
            features_list.append(feat)
        
        # Pad/truncate to expected frames
        padded = []
        for feat in features_list:
            if feat.shape[0] < self.expected_frames:
                pad_size = self.expected_frames - feat.shape[0]
                feat = F.pad(feat, (0, 0, 0, pad_size))
            elif feat.shape[0] > self.expected_frames:
                feat = feat[:self.expected_frames]
            padded.append(feat)
        
        return torch.stack(padded, dim=0).to(self.device)
    
    def extract_embedding(self, audio: np.ndarray, source_sr: int = 44100) -> torch.Tensor:
        """
        从音频提取说话人 embedding
        audio: (samples,) 或 (channels, samples)
        source_sr: 源采样率
        返回: (embedding_size,)
        """
        import scipy.signal
        
        # 转单声道
        if len(audio.shape) == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        
        audio = audio.astype(np.float32)
        
        # 重采样到 16kHz
        if source_sr != self.sample_rate:
            audio = scipy.signal.resample_poly(audio, self.sample_rate, source_sr).astype(np.float32)
        
        # 截取/填充到 2 秒
        if len(audio) > self.segment_samples:
            audio = audio[:self.segment_samples]
        elif len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        
        audio_tensor = torch.from_numpy(audio)
        features = self.extract_fbank(audio_tensor)
        embedding = self.model.infer(features)
        
        return embedding.squeeze(0)
    
    def extract_dominant_embedding(
        self,
        audio: np.ndarray,
        source_sr: int = 44100,
        segment_duration: float = 2.0,
        energy_threshold: float = 0.1,
        max_clusters: int = 3
    ) -> Optional[torch.Tensor]:
        """从长音频提取主导说话人 embedding (带聚类)"""
        import scipy.signal
        from sklearn.cluster import SpectralClustering
        from collections import Counter
        
        # 转单声道
        if len(audio.shape) == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        
        audio = audio.astype(np.float32)
        
        # 重采样到 16kHz
        if source_sr != self.sample_rate:
            audio = scipy.signal.resample_poly(audio, self.sample_rate, source_sr).astype(np.float32)
        
        # 分段
        segment_samples = int(segment_duration * self.sample_rate)
        total_samples = len(audio)
        
        segments = []
        for start in range(0, total_samples, segment_samples):
            end = start + segment_samples
            if end > total_samples:
                break
            
            segment = audio[start:end]
            rms = np.sqrt(np.mean(segment ** 2))
            
            if rms > energy_threshold:
                segments.append(segment)
        
        if len(segments) == 0:
            return None
        
        # 批量提取特征
        audio_batch = torch.from_numpy(np.stack(segments, axis=0))
        features = self.extract_fbank(audio_batch)
        
        # 批量推理
        embeddings = self.model.infer(features)
        embeddings_np = embeddings.cpu().numpy()
        
        # 聚类
        n_clusters = min(max_clusters, len(segments))
        if n_clusters <= 1:
            return embeddings.mean(dim=0)
        
        # 余弦相似度矩阵
        emb_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
        affinity = (np.dot(emb_norm, emb_norm.T) + 1.0) / 2.0
        
        try:
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = clustering.fit_predict(affinity)
        except Exception:
            return torch.from_numpy(embeddings_np.mean(axis=0)).float().to(self.device)
        
        # 最大簇
        label_counts = Counter(labels)
        largest_label = label_counts.most_common(1)[0][0]
        largest_mask = labels == largest_label
        
        mean_emb = embeddings_np[largest_mask].mean(axis=0)
        return torch.from_numpy(mean_emb).float().to(self.device)


def export_campplus(
    model_dir: str,
    output_dir: str,
    batch_size: int = 1,
    fp16: bool = True
):
    """导出 CAMP++ 到 ONNX 和 TensorRT"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    onnx_path = output_dir / f"campplus_b{batch_size}.onnx"
    trt_path = output_dir / f"campplus_b{batch_size}{'_fp16' if fp16 else ''}.trt"
    
    print("=" * 60)
    print("CAMP++ Export")
    print("=" * 60)
    print(f"  Model dir: {model_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  FP16: {fp16}")
    print(f"  Fixed input: 2s audio @ 16kHz -> ({batch_size}, {TIME_FRAMES}, {FEAT_DIM})")
    print("=" * 60)
    
    # 导出 ONNX
    export_campplus_to_onnx(model_dir, str(onnx_path), batch_size)
    
    # 转换 TensorRT
    onnx_to_tensorrt(str(onnx_path), str(trt_path), fp16)
    
    print("\n✓ Export completed!")
    print(f"  ONNX: {onnx_path}")
    print(f"  TRT: {trt_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='./exported_spk')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    
    args = parser.parse_args()
    
    export_campplus(args.model_dir, args.output, args.batch, args.fp16)
