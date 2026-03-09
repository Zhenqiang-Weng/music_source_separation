"""
CAMP++ Model:  A fast and efficient speaker embedding model
Using 2D CNN as head and densely connected TDNN as backbone
"""

import torch
import torch. nn as nn
import torch.nn.functional as F
import torch. utils.checkpoint as cp
from collections import OrderedDict


# ============================================================================
# Layer Components
# ============================================================================

def get_nonlinear(config_str, channels):
    """Create nonlinear activation layers based on config string"""
    nonlinear = nn. Sequential()
    for name in config_str.split('-'):
        if name == 'relu': 
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu': 
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_': 
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f'Unexpected module:  {name}')
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    """Statistics pooling:  concatenate mean and std"""
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """Statistics Pooling Layer"""
    def forward(self, x):
        return statistics_pooling(x)


# ============================================================================
# Basic Building Blocks
# ============================================================================

class BasicResBlock(nn.Module):
    """Basic Residual Block for 2D CNN"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn. Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn. Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=(stride, 1), bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TDNNLayer(nn.Module):
    """Time Delay Neural Network Layer"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f'Expect odd kernel size, got {kernel_size}'
            padding = (kernel_size - 1) // 2 * dilation
        
        self.linear = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """Context Attention Module Layer"""
    def __init__(self, bn_channels, out_channels, kernel_size,
                 stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x. mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        """Segment-level pooling"""
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """CAM Dense TDNN Layer"""
    def __init__(self, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, f'Expect odd kernel size, got {kernel_size}'
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self. memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self. nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """CAM Dense TDNN Block with dense connections"""
    def __init__(self, num_layers, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.add_module(f'tdnnd{i+1}', layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """Transition Layer between blocks"""
    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Dense (Fully Connected) Layer"""
    def __init__(self, in_channels, out_channels, bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


# ============================================================================
# FCM:  Frequency Convolutional Module (Head)
# ============================================================================

class FCM(nn.Module):
    """Frequency Convolutional Module - 2D CNN Head"""
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2],
                 m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3,
            stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn. BatchNorm2d(m_channels)
        
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers. append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn. Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        # Reshape:  (B, C, H, W) -> (B, C*H, W)
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


# ============================================================================
# CAMP++ Main Model
# ============================================================================

class CAMPPlus(nn.Module):
    """
    CAMP++ Speaker Embedding Model
    
    Architecture:
        - Head: FCM (Frequency Convolutional Module) - 2D CNN
        - Backbone:  Densely Connected TDNN with Context Attention
        - Pooling: Statistics Pooling (mean + std)
        - Output: Dense layer for embedding
    
    Args:
        feat_dim: Input feature dimension (default: 80 for fbank)
        embedding_size: Output embedding size (default: 512)
        growth_rate: Growth rate for dense connections (default: 32)
        bn_size: Bottleneck size multiplier (default: 4)
        init_channels: Initial TDNN channels (default: 128)
        config_str:  Nonlinearity configuration (default: 'batchnorm-relu')
        memory_efficient: Use checkpointing to save memory (default: True)
        output_level: 'segment' for utterance-level, 'frame' for frame-level
    """
    
    def __init__(self,
                 feat_dim=80,
                 embedding_size=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True,
                 output_level='segment'):
        super(CAMPPlus, self).__init__()
        
        # Head: 2D CNN for frequency modeling
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level
        
        # Backbone: Dense TDNN blocks
        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn', TDNNLayer(
                    channels, init_channels, kernel_size=5,
                    stride=2, dilation=1, padding=-1, config_str=config_str
                ))
            ])
        )
        
        channels = init_channels
        
        # Add 3 Dense TDNN Blocks with different configurations
        block_configs = [
            (12, 3, 1),  # (num_layers, kernel_size, dilation)
            (24, 3, 2),
            (16, 3, 2)
        ]
        
        for i, (num_layers, kernel_size, dilation) in enumerate(block_configs):
            # Dense TDNN Block
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.xvector.add_module(f'block{i+1}', block)
            
            # Update channels after dense block
            channels = channels + num_layers * growth_rate
            
            # Transition layer to reduce dimensions
            transit = TransitLayer(
                channels, channels // 2, 
                bias=False, config_str=config_str
            )
            self.xvector.add_module(f'transit{i+1}', transit)
            channels //= 2
        
        # Output nonlinearity
        self.xvector.add_module('out_nonlinear', get_nonlinear(config_str, channels))
        
        # Pooling and embedding layer
        if self.output_level == 'segment':
            self.xvector.add_module('stats', StatsPool())
            self.xvector.add_module(
                'dense',
                DenseLayer(channels * 2, embedding_size, config_str='batchnorm_')
            )
        else:
            assert self.output_level == 'frame', \
                "`output_level` should be 'segment' or 'frame'"
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn. Linear)):
                nn.init. kaiming_normal_(m.weight. data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, T, F) where
               B = batch size, T = time steps, F = feature dimension
        
        Returns: 
            Embedding tensor of shape (B, embedding_size) for segment-level
            or (B, T', channels) for frame-level
        """
        # Permute from (B, T, F) to (B, F, T)
        x = x.permute(0, 2, 1)
        
        # 2D CNN head
        x = self.head(x)
        
        # Dense TDNN backbone
        x = self.xvector(x)
        
        # Permute back for frame-level output
        if self.output_level == 'frame':
            x = x. transpose(1, 2)
        
        return x


# ============================================================================
# Model Factory
# ============================================================================

def create_campplus_model(feat_dim=80, embedding_size=512, **kwargs):
    """
    Factory function to create CAMP++ model
    
    Args:
        feat_dim: Input feature dimension
        embedding_size: Output embedding dimension
        **kwargs: Additional arguments for CAMPPlus
    
    Returns:
        CAMPPlus model instance
    """
    return CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size, **kwargs)


if __name__ == '__main__': 
    # Test model
    model = CAMPPlus(feat_dim=80, embedding_size=512)
    
    # Print model info
    print("=" * 70)
    print("CAMP++ Model Architecture")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    # Test forward pass
    batch_size = 2
    seq_len = 200
    feat_dim = 80
    
    x = torch.randn(batch_size, seq_len, feat_dim)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)