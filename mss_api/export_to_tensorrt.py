"""
Export BS-Roformer Core to ONNX/TensorRT
Core model without STFT/iSTFT for compatibility
"""

import torch
import sys
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.bs_roformer.bs_roformer_exportable import BDCSGBSRoformerExportable, convert_full_to_exportable
from utils.settings import get_model_from_config
from utils.model_utils import load_start_checkpoint
from types import SimpleNamespace


def export_onnx(model_path, config_path, output_path, batch_size=1):
    """Export core model to ONNX"""
    print(f"Loading full model: {model_path}")
    
    args = SimpleNamespace()
    args.model_type = 'bdc_sg_bs_roformer'
    args.config_path = config_path
    args.start_check_point = model_path
    
    # Load full model
    full_model, config = get_model_from_config(args.model_type, config_path)
    if model_path:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        load_start_checkpoint(args, full_model, ckpt, type_='inference')
    full_model.eval()
    
    # Create exportable model with same config
    model_cfg = config.model
    
    # 打印配置信息
    print(f"\n=== Model Config ===")
    print(f"  dim: {model_cfg.dim}")
    print(f"  depth: {model_cfg.depth}")
    print(f"  heads: {model_cfg.heads}")
    print(f"  dim_head: {model_cfg.dim_head}")
    print(f"  mask_estimator_depth: {model_cfg.mask_estimator_depth}")
    
    exportable_model = BDCSGBSRoformerExportable(
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        freqs_per_bands=tuple(model_cfg.freqs_per_bands),
        dim_head=model_cfg.dim_head,
        heads=model_cfg.heads,
        attn_dropout=0.,
        ff_dropout=0.,
        flash_attn=False,
        mask_estimator_depth=model_cfg.mask_estimator_depth,
        mlp_expansion_factor=getattr(model_cfg, 'mlp_expansion_factor', 4),
        skip_connection=getattr(model_cfg, 'skip_connection', False),
    )
    
    # Copy weights with verbose output
    print("\nCopying weights to exportable model...")
    exportable_model = convert_full_to_exportable(full_model, exportable_model, verbose=True)
    exportable_model.eval()
    
    # 从 config 获取正确的 time_frames
    chunk_size = config.audio.chunk_size
    hop_length = model_cfg.stft_hop_length
    n_fft = model_cfg.stft_n_fft
    
    # 使用 center=True 时的计算公式：time_frames = floor(samples / hop_length) + 1
    time_frames = chunk_size // hop_length + 1
    freq_bins = n_fft // 2 + 1  # 1025
    spk_dim = 192
    
    print(f"\n=== STFT Config ===")
    print(f"  chunk_size: {chunk_size}")
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    print(f"  time_frames (center=True): {time_frames}")
    print(f"  freq_bins: {freq_bins}")
    
    # 验证：实际做一次 STFT
    test_audio = torch.randn(1, 2, chunk_size)
    test_stft = torch.stft(
        test_audio.reshape(2, chunk_size),
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=torch.hann_window(n_fft), center=True, return_complex=True
    )
    actual_time_frames = test_stft.shape[-1]
    print(f"  Verified time_frames: {actual_time_frames}")
    
    if actual_time_frames != time_frames:
        print(f"  ⚠️ Adjusting time_frames from {time_frames} to {actual_time_frames}")
        time_frames = actual_time_frames
    
    # Input shapes
    dummy_stft = torch.randn(batch_size, freq_bins * 2, time_frames, 2, dtype=torch.float32)
    dummy_spk = torch.randn(batch_size, spk_dim, dtype=torch.float32)
    
    print(f"\nSTFT input shape: {dummy_stft.shape}")
    print(f"Speaker embedding shape: {dummy_spk.shape}")
    
    # Test forward
    with torch.no_grad():
        output = exportable_model(dummy_stft, dummy_spk)
    print(f"Output shape: {output.shape}")
    
    # 验证输出不是全零
    print(f"\nOutput stats:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Max: {output.abs().max().item():.6f}")
    
    if output.std().item() < 1e-6:
        print("\n⚠️ WARNING: Output is nearly zero! Model weights may not be loaded correctly.")
    
    # Export
    print(f"\nExporting to: {output_path}")
    torch.onnx.export(
        exportable_model,
        (dummy_stft, dummy_spk),
        output_path,
        input_names=['stft_features', 'speaker_embedding'],
        output_names=['masked_stft'],
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ ONNX exported: {output_path}")
    return output_path, time_frames


def onnx_to_tensorrt(onnx_path, output_path, fp16=False, bf16=False, workspace_gb=16):
    """Convert ONNX to TensorRT with PyTorch Autocast-like mixed precision"""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed")
        return None
    
    print(f"\nConverting to TensorRT...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {output_path}")
    print(f"  FP16: {fp16}")
    print(f"  BF16: {bf16}")
    print(f"  Workspace: {workspace_gb}GB")
    
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
    
    # 使用基础策略
    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT)
    )
    
    # 启用FP16/BF16
    use_reduced_precision = fp16 or bf16
    
    if use_reduced_precision and builder.platform_has_fast_fp16:
        # 检查是否支持 OBEY_PRECISION_CONSTRAINTS
        has_obey_precision = hasattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS')
        
        if bf16 and hasattr(trt.BuilderFlag, 'BF16'):
            config.set_flag(trt.BuilderFlag.BF16)
            print("✓ BF16 enabled")
            precision_type = trt.DataType.BF16
        elif fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 enabled")
            precision_type = trt.DataType.HALF
        else:
            precision_type = None
        
        if precision_type:
            if has_obey_precision:
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                print("  Using OBEY_PRECISION_CONSTRAINTS")
            
            # 不能设置精度的层类型
            skip_layer_types = set()
            for type_name in ['SHAPE', 'CONSTANT', 'CONCATENATION', 'GATHER', 'SLICE', 'SHUFFLE']:
                if hasattr(trt.LayerType, type_name):
                    skip_layer_types.add(getattr(trt.LayerType, type_name))
            
            # 需要保持FP32的敏感层类型
            fp32_layer_types = set()
            for type_name in ['SOFTMAX', 'REDUCE', 'NORMALIZATION', 'UNARY']:
                if hasattr(trt.LayerType, type_name):
                    fp32_layer_types.add(getattr(trt.LayerType, type_name))
            
            # FP16安全的操作
            fp16_safe_ops = set()
            for type_name in ['MATRIX_MULTIPLY', 'CONVOLUTION', 'FULLY_CONNECTED']:
                if hasattr(trt.LayerType, type_name):
                    fp16_safe_ops.add(getattr(trt.LayerType, type_name))
            
            # 必须使用FP32的层名称关键词（模型特定）
            # band_split 和 final_norm 强制 FP32
            fp32_layer_keywords = [
                'band_split',      # self.band_split
                'final_norm',      # self.final_norm
                'norm',            # 所有 norm 层
                'softmax', 
                'reduce', 
                'sqrt', 
                'exp', 
                'div', 
                'pow', 
                'layernorm', 
                'rmsnorm'
            ]
            
            fp32_count = 0
            fp16_count = 0
            skip_count = 0
            
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                layer_type = layer.type
                layer_name = layer.name.lower()
                
                # 跳过不能设置精度的层
                if layer_type in skip_layer_types:
                    skip_count += 1
                    continue
                
                # 检查是否是数值敏感操作（层类型或层名称匹配）
                is_sensitive = (
                    layer_type in fp32_layer_types or
                    any(keyword in layer_name for keyword in fp32_layer_keywords)
                )
                
                try:
                    if is_sensitive:
                        # 强制使用FP32
                        layer.precision = trt.DataType.FLOAT
                        for j in range(layer.num_outputs):
                            layer.set_output_type(j, trt.DataType.FLOAT)
                        fp32_count += 1
                    elif layer_type in fp16_safe_ops:
                        # 允许使用低精度
                        layer.precision = precision_type
                        for j in range(layer.num_outputs):
                            layer.set_output_type(j, precision_type)
                        fp16_count += 1
                except Exception:
                    skip_count += 1
                    continue
            
            print(f"  Mixed Precision Strategy (PyTorch Autocast-like):")
            print(f"    FP32 layers: {fp32_count} (sensitive ops including band_split, final_norm)")
            print(f"    FP16/BF16 layers: {fp16_count} (safe ops)")
            print(f"    Skipped layers: {skip_count} (shape/constant ops)")
            print(f"    Auto layers: {network.num_layers - fp32_count - fp16_count - skip_count}")
    else:
        print("✓ Using FP32 (full precision)")
    
    print("Building TensorRT engine (this may take a while)...")
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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='./exported')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='Enable FP16')
    parser.add_argument('--bf16', action='store_true', help='Enable BF16 (recommended, better stability)')
    parser.add_argument('--workspace', type=int, default=16)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    model_name = Path(args.model).stem
    onnx_path = output_dir / f"{model_name}_core_b{args.batch}.onnx"
    
    # 文件名根据精度类型
    precision_suffix = '_bf16' if args.bf16 else ('_fp16' if args.fp16 else '')
    trt_path = output_dir / f"{model_name}_core_b{args.batch}{precision_suffix}.trt"
    
    print("=" * 60)
    print("BS-Roformer Core Export (No STFT/iSTFT)")
    print("=" * 60)
    
    onnx_file, time_frames = export_onnx(args.model, args.config, str(onnx_path), args.batch)
    
    print(f"\n=== Expected shapes ===")
    print(f"STFT Input: (batch={args.batch}, 2050, {time_frames}, 2)")
    print(f"Speaker Input: (batch={args.batch}, 192)")
    print(f"Output: (batch={args.batch}, 3, 2050, {time_frames}, 2)")
    
    onnx_to_tensorrt(str(onnx_path), str(trt_path), args.fp16, args.bf16, args.workspace)
    
    print("\n✓ Export completed!")
