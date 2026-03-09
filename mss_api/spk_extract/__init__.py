"""
CAMP++ Speaker Verification Model
Complete package with model definition, checkpoint management, and initialization

Usage:
    # Quick load from checkpoint
    >>> from campplus import load_model
    >>> model = load_model('./checkpoint')
    
    # Create and save checkpoint
    >>> from campplus import CAMPPlus, save_checkpoint
    >>> model = CAMPPlus(feat_dim=80, embedding_size=512)
    >>> save_checkpoint(model, './checkpoint')
    
    # Advanced usage
    >>> from campplus import init_from_checkpoint, create_checkpoint
    >>> model = init_from_checkpoint('./checkpoint', device='cuda')
"""

import os
import json
import torch
import torchaudio.compliance.kaldi as Kaldi
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, NamedTuple
import scipy.signal
from sklearn.cluster import SpectralClustering
from collections import Counter

# ============================================================================
# Import Core Model Components
# ============================================================================

from .campplus_model import (
    CAMPPlus,
    FCM,
    TDNNLayer,
    CAMLayer,
    CAMDenseTDNNLayer,
    CAMDenseTDNNBlock,
    TransitLayer,
    DenseLayer,
    StatsPool,
    BasicResBlock,
    get_nonlinear,
    statistics_pooling,
    create_campplus_model
)

# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """Manages model checkpoints with configuration files"""
    
    CONFIG_FILE = "configuration.json"
    MODEL_FILE = "campplus_cn_en_common.pt"
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def save(self, model: torch.nn.Module, config: Dict[str, Any], 
             filename: Optional[str] = None) -> str:
        """Save model checkpoint with configuration"""
        self. checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = self.MODEL_FILE
            
        # Save model state dict
        model_path = self.checkpoint_dir / filename
        torch.save(model.state_dict(), model_path)
        
        # Save configuration
        config_path = self. checkpoint_dir / self.CONFIG_FILE
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Config saved to: {config_path}")
        return str(model_path)
    
    def load(self, filename: Optional[str] = None, 
             device: str = 'cpu') -> tuple:
        """Load model checkpoint and configuration"""
        if filename is None:
            filename = self.MODEL_FILE
            
        # Load configuration
        config_path = self. checkpoint_dir / self.CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration not found: {config_path}\n"
                f"Expected checkpoint structure:\n"
                f"  {self.checkpoint_dir}/\n"
                f"  ├── configuration.json\n"
                f"  └── {filename}"
            )
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Load model state dict
        model_path = self.checkpoint_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        state_dict = torch.load(model_path, map_location=device)
        
        print(f"✓ Loaded from: {self.checkpoint_dir}")
        return state_dict, config
    
    def exists(self) -> bool:
        """Check if checkpoint exists"""
        config_exists = (self.checkpoint_dir / self. CONFIG_FILE).exists()
        model_exists = (self.checkpoint_dir / self.MODEL_FILE).exists()
        return config_exists and model_exists


# ============================================================================
# Configuration Management
# ============================================================================

def create_config(feat_dim: int = 80,
                 embedding_size: int = 512,
                 growth_rate:  int = 32,
                 bn_size: int = 4,
                 init_channels:  int = 128,
                 config_str: str = 'batchnorm-relu',
                 memory_efficient: bool = True,
                 output_level: str = 'segment',
                 **kwargs) -> Dict[str, Any]:
    """
    Create model configuration dictionary
    
    Args: 
        feat_dim: Input feature dimension (default: 80)
        embedding_size:  Output embedding dimension (default: 512)
        growth_rate: Dense block growth rate (default: 32)
        bn_size: Bottleneck size multiplier (default: 4)
        init_channels: Initial TDNN channels (default: 128)
        config_str: Activation configuration (default: 'batchnorm-relu')
        memory_efficient: Use gradient checkpointing (default: True)
        output_level: 'segment' or 'frame' (default: 'segment')
        **kwargs: Additional metadata
        
    Returns:
        Configuration dictionary
    """
    config = {
        "model_type": "campplus",
        "model_version":  "1.0",
        "framework": "pytorch",
        
        "model_config": {
            "feat_dim": feat_dim,
            "embedding_size": embedding_size,
            "growth_rate": growth_rate,
            "bn_size": bn_size,
            "init_channels": init_channels,
            "config_str": config_str,
            "memory_efficient":  memory_efficient,
            "output_level": output_level
        },
        
        "feature_config": {
            "sample_rate": kwargs.get("sample_rate", 16000),
            "feature_type": "fbank",
            "num_mel_bins": feat_dim,
            "frame_length": kwargs.get("frame_length", 25),
            "frame_shift": kwargs.get("frame_shift", 10),
        },
        
        "training_info": {
            "trained_on": kwargs.get("trained_on", "unknown"),
            "num_speakers": kwargs.get("num_speakers", -1),
            "num_epochs": kwargs.get("num_epochs", -1),
            "description": kwargs.get("description", "")
        }
    }
    
    return config


# ============================================================================
# Main Initialization Functions
# ============================================================================

def init_from_checkpoint(model_dir: str,
                        device: str = 'cpu',
                        pretrained_model:  str = 'campplus_cn_en_common.pt',
                        strict: bool = True) -> CAMPPlus:
    """
    Initialize CAMP++ model from checkpoint directory
    
    This is the main initialization function that mimics ModelScope's pattern. 
    It loads both the configuration and model weights from a checkpoint directory.
    
    Args:
        model_dir: Directory containing configuration. json and model checkpoint
        device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
        pretrained_model: Checkpoint filename (default: 'campplus_cn_en_common.pth')
        strict: Strict mode for loading state dict (default: True)
        
    Returns:
        Initialized and loaded CAMPPlus model
        
    Example:
        >>> model = init_from_checkpoint('./my_checkpoint', device='cuda')
        >>> features = torch.randn(1, 200, 80)
        >>> embedding = model(features)
        
    Raises:
        FileNotFoundError: If checkpoint directory or files don't exist
        RuntimeError: If model loading fails
    """
    # Load checkpoint
    manager = CheckpointManager(model_dir)
    state_dict, config = manager.load(filename=pretrained_model, device=device)
    
    # Extract model config - handle different config structures
    model_config = config.get("model_config", {})
    
    # If model_config is a string (path to yaml), look for nested config
    if isinstance(model_config, str) or not model_config:
        # Try nested structure: config["model"]["model_config"]
        nested_config = config.get("model", {}).get("model_config", {})
        if nested_config and isinstance(nested_config, dict):
            # Map the nested config keys to CAMPPlus expected keys
            model_config = {
                "feat_dim": nested_config.get("fbank_dim", 80),
                "embedding_size": nested_config.get("emb_size", 192),
            }
        else:
            raise ValueError(
                f"Invalid configuration file. Missing valid 'model_config' section.\n"
                f"Config keys found: {list(config.keys())}"
            )
    
    # Create model
    try:
        model = CAMPPlus(**model_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create model with config {model_config}: {e}")
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")
    
    # Setup model
    model.to(device)
    model.eval()
    
    # Print info
    feat_dim = model_config.get('feat_dim', 'unknown')
    emb_size = model_config.get('embedding_size', 'unknown')
    print(f"✓ CAMP++ Model Ready")
    print(f"  - Feature dim: {feat_dim}")
    print(f"  - Embedding size: {emb_size}")
    print(f"  - Device: {device}")
    
    return model


def save_checkpoint(model: CAMPPlus,
                   checkpoint_dir: str,
                   config: Optional[Dict[str, Any]] = None,
                   filename: Optional[str] = None,
                   **config_kwargs) -> str:
    """
    Save CAMP++ model to checkpoint directory
    
    Args:
        model: CAMPPlus model instance to save
        checkpoint_dir: Directory to save checkpoint
        config: Optional configuration dict (auto-generated if None)
        filename: Optional checkpoint filename (default: 'campplus.pth')
        **config_kwargs: Config parameters if auto-generating config
        
    Returns:
        Path to saved checkpoint file
        
    Example:
        >>> model = CAMPPlus(feat_dim=80, embedding_size=512)
        >>> save_checkpoint(model, './my_checkpoint', 
        ...                 trained_on='VoxCeleb', num_epochs=100)
    """
    manager = CheckpointManager(checkpoint_dir)
    
    # Auto-generate config if not provided
    if config is None:
        # Try to infer parameters from model
        try:
            feat_dim = config_kwargs.get('feat_dim', 80)
            embedding_size = config_kwargs.get('embedding_size', 512)
            config = create_config(feat_dim=feat_dim, 
                                 embedding_size=embedding_size,
                                 **config_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create config: {e}")
    
    return manager.save(model, config, filename=filename)


def create_checkpoint(checkpoint_dir: str,
                     feat_dim: int = 80,
                     embedding_size: int = 512,
                     random_init: bool = True,
                     **kwargs) -> CAMPPlus:
    """
    Create a new CAMP++ model and save as checkpoint
    
    Args: 
        checkpoint_dir: Directory to save checkpoint
        feat_dim: Input feature dimension
        embedding_size:  Output embedding dimension
        random_init: Use random initialization (default: True)
        **kwargs: Additional model/config parameters
        
    Returns: 
        Created model instance
        
    Example: 
        >>> model = create_checkpoint('./new_checkpoint', feat_dim=80, 
        ...                          embedding_size=512)
    """
    # Create model
    model_kwargs = {
        'feat_dim': feat_dim,
        'embedding_size': embedding_size,
        'growth_rate': kwargs.pop('growth_rate', 32),
        'bn_size': kwargs.pop('bn_size', 4),
        'init_channels': kwargs.pop('init_channels', 128),
        'config_str': kwargs.pop('config_str', 'batchnorm-relu'),
        'memory_efficient': kwargs.pop('memory_efficient', True),
        'output_level': kwargs.pop('output_level', 'segment')
    }
    
    model = CAMPPlus(**model_kwargs)
    
    # Save checkpoint
    config = create_config(**model_kwargs, **kwargs)
    manager = CheckpointManager(checkpoint_dir)
    manager.save(model, config)
    
    print(f"✓ New checkpoint created at: {checkpoint_dir}")
    return model


# ============================================================================
# Convenient Loading Functions
# ============================================================================

def load_model(checkpoint_dir:  str, 
               device: Optional[str] = None,
               **kwargs) -> CAMPPlus:
    """
    Simple one-line model loading (auto device detection)
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load on (auto-detect if None)
        **kwargs: Additional arguments for init_from_checkpoint
        
    Returns:
        Loaded CAMPPlus model
        
    Example:
        >>> model = load_model('./checkpoint')  # Auto device
        >>> model = load_model('./checkpoint', device='cuda')  # Specific device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return init_from_checkpoint(checkpoint_dir, device=device, **kwargs)


def load_model_cpu(checkpoint_dir: str, **kwargs) -> CAMPPlus:
    """Load model on CPU"""
    return init_from_checkpoint(checkpoint_dir, device='cpu', **kwargs)


def load_model_cuda(checkpoint_dir: str, gpu_id: int = 0, **kwargs) -> CAMPPlus:
    """Load model on CUDA"""
    device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cuda'
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = 'cpu'
    return init_from_checkpoint(checkpoint_dir, device=device, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def check_checkpoint(checkpoint_dir: str) -> bool:
    """
    Check if a valid checkpoint exists
    
    Args: 
        checkpoint_dir: Path to checkpoint directory
        
    Returns: 
        True if valid checkpoint exists, False otherwise
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.exists()


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get checkpoint configuration information
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> info = get_checkpoint_info('./checkpoint')
        >>> print(f"Embedding size: {info['model_config']['embedding_size']}")
    """
    config_path = Path(checkpoint_dir) / "configuration.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Audio Loading (soundfile version - ModelScope style)
# ============================================================================

def load_audio_soundfile(audio_path: str,
                         target_sr:  int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio using soundfile (ModelScope implementation)
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        audio: Audio waveform numpy array (samples,)
        sr: Sample rate
    
    Example:
        >>> audio, sr = load_audio_soundfile('speaker1.wav')
        >>> print(f"Audio shape: {audio.shape}, SR: {sr}")
    """
    # Read audio file
    audio, sr = sf.read(audio_path, dtype='float32')
    
    # Convert stereo to mono (take first channel)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    
    # Resample if needed
    if sr != target_sr:
        import scipy.signal
        audio = scipy.signal.resample_poly(
            audio, target_sr, sr
        ).astype('float32')
        sr = target_sr
    
    return audio, sr


# ============================================================================
# Feature Extraction (Fbank)
# ============================================================================

def extract_fbank(audio: Union[torch.Tensor, np.ndarray],
                 sample_rate:  int = 16000,
                 num_mel_bins: int = 80,
                 frame_length: float = 25.0,
                 frame_shift: float = 10.0,
                 apply_cmvn: bool = True) -> torch.Tensor:
    """
    Extract Fbank (Filter Bank) features from audio
    CAMP++ 在 ModelScope 中的特征提取方法
    
    Args:
        audio: Audio waveform (samples,) - numpy array or torch tensor
        sample_rate:  Sample rate (default: 16000 Hz)
        num_mel_bins: Number of mel filter banks (default: 80)
        frame_length: Frame length in milliseconds (default: 25.0)
        frame_shift:  Frame shift in milliseconds (default:  10.0)
        apply_cmvn: Apply Cepstral Mean Normalization (default: True)
    
    Returns:
        features: Fbank features, shape (time_steps, num_mel_bins)
    
    Example:
        >>> audio, sr = load_audio_soundfile('test.wav')
        >>> features = extract_fbank(audio, sample_rate=sr)
        >>> print(f"Features shape: {features.shape}")  # (T, 80)
    """
    # Convert numpy to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    # Ensure 2D shape:  (1, samples) for Kaldi. fbank
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    # Extract Fbank features using Kaldi
    # This is the EXACT method used in CAMP++ model
    features = Kaldi.fbank(
        audio,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        frame_length=frame_length,
        frame_shift=frame_shift
    )
    
    # Apply Cepstral Mean Normalization (CMVN)
    # 减去均值，这是 CAMP++ 使用的归一化方法
    if apply_cmvn:
        features = features - features.mean(dim=0, keepdim=True)
    
    return features


# ============================================================================
# Complete Pipeline Function
# ============================================================================

def load_and_extract_features(audio_path: str,
                             target_sr: int = 16000,
                             num_mel_bins: int = 80,
                             apply_cmvn: bool = True) -> torch.Tensor:
    """
    一步完成：从音频文件到特征提取
    
    Args:
        audio_path:  Path to audio file
        target_sr: Target sample rate
        num_mel_bins: Number of mel bins
        apply_cmvn:  Apply mean normalization
    
    Returns: 
        features: Fbank features (time_steps, num_mel_bins)
    
    Example:
        >>> features = load_and_extract_features('audio. wav')
        >>> print(features.shape)  # torch.Size([T, 80])
    """
    # Step 1: Load audio
    audio, sr = load_audio_soundfile(audio_path, target_sr=target_sr)
    
    # Step 2: Extract features
    features = extract_fbank(
        audio,
        sample_rate=sr,
        num_mel_bins=num_mel_bins,
        apply_cmvn=apply_cmvn
    )
    
    return features


def extract_embedding(model: CAMPPlus, 
                     features: Union[torch.Tensor, Any],
                     device: Optional[str] = None) -> torch.Tensor:
    """
    Extract speaker embedding from features
    
    Args:
        model: CAMPPlus model
        features: Input features (batch, time, feat_dim) or convertible to tensor
        device: Device for computation (use model's device if None)
        
    Returns:
        Speaker embeddings (batch, embedding_size)
        
    Example:
        >>> model = load_model('./checkpoint')
        >>> features = torch.randn(2, 200, 80)
        >>> embeddings = extract_embedding(model, features)
        >>> print(embeddings.shape)  # (2, 512)
    """
    model.eval()
    
    # Convert to tensor if needed
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    
    # Ensure correct shape
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
    
    assert len(features.shape) == 3, \
        f"Expected shape (batch, time, feat_dim), got {features.shape}"
    
    # Get device
    if device is None: 
        device = next(model.parameters()).device
    
    # Extract embedding
    with torch.no_grad():
        features = features.to(device)
        embeddings = model(features)
    
    return embeddings.cpu()



# ============================================================================
# Core Similarity Computation (ModelScope Implementation)
# ============================================================================

def compute_cos_similarity(emb1: Union[torch.Tensor, np.ndarray],
                          emb2: Union[torch. Tensor, np.ndarray]) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Implementation from ModelScope speaker verification pipeline: 
    - Uses torch.nn.CosineSimilarity with dim=1 and eps=1e-6
    - Returns a single float value representing similarity in range [-1, 1]
    
    Mathematical Formula:
        cos_sim = (emb1 · emb2) / (||emb1|| * ||emb2||)
    
    Args:
        emb1: First embedding, shape (embedding_dim,) or (1, embedding_dim)
        emb2: Second embedding, shape (embedding_dim,) or (1, embedding_dim)
    
    Returns:
        Cosine similarity score in range [-1, 1]
        - 1.0: Identical embeddings
        - 0.0: Orthogonal embeddings
        - -1.0: Opposite embeddings
    
    Example:
        >>> emb1 = torch.randn(512)
        >>> emb2 = torch.randn(512)
        >>> score = compute_cos_similarity(emb1, emb2)
        >>> print(f"Similarity: {score:.4f}")
    """
    # Convert numpy to tensor if needed
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)
    
    # Ensure 2D shape:  (batch_size, embedding_dim)
    if len(emb1.shape) == 1:
        emb1 = emb1.unsqueeze(0)
    if len(emb2.shape) == 1:
        emb2 = emb2.unsqueeze(0)
    
    # Validate shapes
    assert len(emb1.shape) == 2 and len(emb2.shape) == 2, \
        f"Expected 2D tensors, got shapes {emb1.shape} and {emb2.shape}"
    
    # Create cosine similarity function
    # dim=1: compute similarity along embedding dimension
    # eps=1e-6: small value to avoid division by zero
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # Compute cosine similarity
    cosine = cos(emb1, emb2)
    
    # Return scalar value
    return cosine.item()


# ============================================================================
# Dominant Speaker Embedding Extraction with Spectral Clustering
# ============================================================================

class ClusteringResult(NamedTuple):
    mean_embedding: torch.Tensor
    all_embeddings: np.ndarray
    labels: np.ndarray
    largest_cluster_label: int
    segment_indices: List[Tuple[int, int]]
    n_clusters: int
    max_inter_cluster_distance: float  


def estimate_k_by_eigengap(affinity_matrix, k_max=20, eps=1e-12):
    W = np.asarray(affinity_matrix, dtype=np.float64)
    n = W.shape[0]
    assert W.shape[0] == W.shape[1], "affinity_matrix must be square"

    W = 0.5 * (W + W.T)
    W = np.maximum(W, 0.0)

    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]

    L = np.eye(n) - S

    m = min(n, k_max + 1)
    evals = np.linalg.eigvalsh(L)[:m]   # 已排序（升序）
    gaps = np.diff(evals)               # gaps[i] = evals[i+1]-evals[i]

    k = int(np.argmax(gaps[:m-1]) + 1)  # +1 because i -> k=i+1
    return k, evals, gaps



def extract_dominant_speaker_embedding_with_clusters(
    model: CAMPPlus,
    audio: np.ndarray,
    source_sr: int = 44100,
    target_sr: int = 16000,
    segment_duration: float = 2.0,
    energy_threshold: float = 0.1,
    max_clusters: int = 3,
    device: Optional[str] = None,
    fine_grain_window: float = 0.5  # 细粒度窗口，默认0.5秒
) -> Optional[ClusteringResult]:

    model.eval()

    # Step 0: 双声道转单声道
    if len(audio.shape) == 2:
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)
        elif audio.shape[1] == 2:
            audio = audio.mean(axis=1)
        else:
            if audio.shape[0] < audio.shape[1]:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    
    # Step 1: 重采样
    if source_sr != target_sr:
        audio = scipy.signal.resample_poly(
            audio, target_sr, source_sr
        ).astype(np.float32)
    
    # Step 2: 细粒度能量检测，提取有效音频段
    fine_window_samples = int(fine_grain_window * target_sr)
    total_samples = len(audio)
    
    # 2.1 标记每个细粒度窗口的能量
    valid_mask = []
    for start_idx in range(0, total_samples, fine_window_samples):
        end_idx = min(start_idx + fine_window_samples, total_samples)
        segment = audio[start_idx:end_idx]
        rms_energy = np.sqrt(np.mean(segment ** 2))
        valid_mask.append(rms_energy > energy_threshold)
    
    # 2.2 合并连续的有效段
    active_segments = []
    i = 0
    while i < len(valid_mask):
        if valid_mask[i]:
            start = i * fine_window_samples
            while i < len(valid_mask) and valid_mask[i]:
                i += 1
            end = min(i * fine_window_samples, total_samples)
            active_segments.append((start, end))
        else:
            i += 1
    
    if len(active_segments) == 0:
        return None
    
    # Step 3: 处理有效段 - 短段循环填充至2s，长段切分为2s
    target_samples = int(segment_duration * target_sr)
    processed_segments = []
    segment_indices = []
    
    for start, end in active_segments:
        segment = audio[start:end]
        seg_len = end - start
        
        if seg_len < target_samples:
            # 短段：循环填充至2s
            num_repeats = int(np.ceil(target_samples / seg_len))
            padded = np.tile(segment, num_repeats)[:target_samples]
            processed_segments.append(padded)
            segment_indices.append((start, end))
        else:
            # 长段：切分为多个2s段
            for seg_start in range(0, seg_len, target_samples):
                seg_end = min(seg_start + target_samples, seg_len)
                chunk = segment[seg_start:seg_end]
                
                # 最后一个片段如果不足2s，也循环填充
                if len(chunk) < target_samples:
                    num_repeats = int(np.ceil(target_samples / len(chunk)))
                    chunk = np.tile(chunk, num_repeats)[:target_samples]
                
                processed_segments.append(chunk)
                segment_indices.append((start + seg_start, start + seg_end))
    
    if len(processed_segments) == 0:
        return None
    
    # Step 4: 批量提取特征和embedding
    features_list = []
    for segment in processed_segments:
        feat = extract_fbank(segment, sample_rate=target_sr)
        features_list.append(feat)
    
    features_batch = torch.stack(features_list, dim=0).to(device)

    with torch.no_grad():
        embeddings = model(features_batch)
    
    embeddings_np = embeddings.cpu().numpy()
    
    # Step 5: 谱聚类
    n_segments = len(processed_segments)
    
    if n_segments < 2:
        mean_embedding = torch.from_numpy(embeddings_np.mean(axis=0)).float()
        labels = np.zeros(n_segments, dtype=int)
        return ClusteringResult(
            mean_embedding=mean_embedding,
            all_embeddings=embeddings_np,
            labels=labels,
            largest_cluster_label=0,
            segment_indices=segment_indices,
            n_clusters=1,
            max_inter_cluster_distance=0.0
        )
    
    # 计算余弦相似度矩阵
    embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    affinity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    affinity_matrix = (affinity_matrix + 1) / 2

    # 谱聚类 - 估计聚类数
    estimated_k, evals, gaps = estimate_k_by_eigengap(affinity_matrix, k_max=min(max_clusters, n_segments-1))
    n_clusters = min(estimated_k, max_clusters, n_segments)
    n_clusters = max(1, n_clusters)
    
    if n_clusters == 1:
        mean_embedding = torch.from_numpy(embeddings_np.mean(axis=0)).float()
        labels = np.zeros(n_segments, dtype=int)
        return ClusteringResult(
            mean_embedding=mean_embedding,
            all_embeddings=embeddings_np,
            labels=labels,
            largest_cluster_label=0,
            segment_indices=segment_indices,
            n_clusters=1,
            max_inter_cluster_distance=0.0
        )
    
    # 执行谱聚类
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(affinity_matrix)
    
    # 找到最大簇
    label_counts = Counter(labels)
    largest_cluster_label = label_counts.most_common(1)[0][0]
    
    # 获取最大簇的embedding均值
    largest_cluster_mask = (labels == largest_cluster_label)
    largest_cluster_embeddings = embeddings_np[largest_cluster_mask]
    mean_embedding = torch.from_numpy(largest_cluster_embeddings.mean(axis=0)).float()
    
    # 计算簇中心之间的最大距离
    cluster_centers = []
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        if cluster_mask.sum() > 0:
            cluster_center = embeddings_np[cluster_mask].mean(axis=0)
            cluster_centers.append(cluster_center)
    
    max_inter_cluster_distance = 0.0
    if len(cluster_centers) > 1:
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                max_inter_cluster_distance = max(max_inter_cluster_distance, dist)
    
    return ClusteringResult(
        mean_embedding=mean_embedding,
        all_embeddings=embeddings_np,
        labels=labels,
        largest_cluster_label=largest_cluster_label,
        segment_indices=segment_indices,
        n_clusters=n_clusters,
        max_inter_cluster_distance=max_inter_cluster_distance
    )

def extract_dominant_speaker_embedding(
    model: CAMPPlus,
    audio: np.ndarray,
    source_sr: int = 44100,
    target_sr: int = 16000,
    segment_duration: float = 2.0,
    energy_threshold: float = 0.1,
    max_clusters: int = 3,
    device: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    从44.1kHz音频中提取主导说话人的embedding
    
    这是简化版本，只返回最大簇的平均embedding。
    如需完整聚类信息，请使用 extract_dominant_speaker_embedding_with_clusters
    """
    result = extract_dominant_speaker_embedding_with_clusters(
        model=model,
        audio=audio,
        source_sr=source_sr,
        target_sr=target_sr,
        segment_duration=segment_duration,
        energy_threshold=energy_threshold,
        max_clusters=max_clusters,
        device=device
    )
    
    if result is None:
        return None
    
    return result.mean_embedding


# ============================================================================
# Package Information
# ============================================================================

__version__ = '1.0.0'
__author__ = 'CAMP++ Implementation'
__all__ = [
    # Main functions
    'init_from_checkpoint',
    'save_checkpoint',
    'create_checkpoint',
    'load_model',
    'load_model_cpu',
    'load_model_cuda',
    
    # Model classes
    'CAMPPlus',
    'create_campplus_model',
    
    # Utilities
    'check_checkpoint',
    'get_checkpoint_info',
    'extract_embedding',
    'create_config',
    'compute_cos_similarity',
    'load_and_extract_features',
    'load_audio_soundfile',
    'extract_fbank',
    'extract_dominant_speaker_embedding',
    'extract_dominant_speaker_embedding_with_clusters',  # 新增
    'ClusteringResult',  # 新增
    
    # Components (for advanced users)
    'FCM',
    'TDNNLayer',
    'CAMLayer',
    'CAMDenseTDNNLayer',
    'CAMDenseTDNNBlock',
    'TransitLayer',
    'DenseLayer',
    'StatsPool',
    'BasicResBlock',
    'get_nonlinear',
    'statistics_pooling',
    
    # Checkpoint manager
    'CheckpointManager',
]


# ============================================================================
# Module-level convenience
# ============================================================================

def info():
    """Print package information"""
    print("=" * 70)
    print("CAMP++ Speaker Verification Model")
    print("=" * 70)
    print(f"Version: {__version__}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print("=" * 70)
    print("\nQuick Start:")
    print("  from campplus import load_model")
    print("  model = load_model('./checkpoint')")
    print("\nDocumentation:")
    print("  help(load_model)")
    print("  help(save_checkpoint)")
    print("=" * 70)


# Show info on import if in interactive mode
if __name__ != '__main__':
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        info()