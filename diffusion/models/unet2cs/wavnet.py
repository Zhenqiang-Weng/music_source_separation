import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_add_tanh_sigmoid_multiply(x, g):
    """
    x: [B, 2*C, T]
    g: [B, 2*C, T]
    returns: acts [B, C, T] after gated activation
    """
    a = x + g
    a1, a2 = a.chunk(2, dim=1)
    return torch.tanh(a1) * torch.sigmoid(a2)


class ResidualGatedBlock(nn.Module):
    """
    Single residual gated block used by WaveNet:
      - dilated convs producing 2*hidden channels (filter & gate)
      - optional conditioning (adds to pre-activations)
      - project to residual + skip via 1x1 conv
    """
    def __init__(self, hidden_channels, kernel_size=3, dilation=1, cond_channels=None, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        # conv that produces filter & gate (2 * hidden)
        self.conv = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                              padding=pad, dilation=dilation, bias=False)
        # projection from gated activation to (residual + skip)
        self.res_skip = nn.Conv1d(hidden_channels, hidden_channels + hidden_channels, kernel_size=1)
        # conditioning projection: if cond_channels provided, map to 2*hidden per layer
        if cond_channels is not None:
            self.cond_proj = nn.Conv1d(cond_channels, 2 * hidden_channels, kernel_size=1, bias=False)
        else:
            self.cond_proj = None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, cond_slice=None):
        """
        x: [B, hidden_channels, T]
        cond_slice: [B, 2*hidden_channels, T] or None
        returns:
          res: [B, hidden_channels, T]   (to add back to x)
          skip: [B, hidden_channels, T]  (to accumulate)
        """
        x_in = self.conv(x)  # [B, 2*hidden, T]
        if self.cond_proj is not None and cond_slice is not None:
            # cond_slice is pre-projected slice for this layer
            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_slice)
        else:
            # split and apply gated activation without external cond
            a1, a2 = x_in.chunk(2, dim=1)
            acts = torch.tanh(a1) * torch.sigmoid(a2)

        acts = self.dropout(acts)
        res_skip = self.res_skip(acts)  # [B, 2*hidden, T]
        res, skip = res_skip.split([x.shape[1], x.shape[1]], dim=1)
        return res, skip


class WaveNet(nn.Module):
    """
    WaveNet-style denoiser compatible with how the repo constructs it:
      WaveNet(in_channels, n_layers, n_chans, n_hidden)
    - in_channels: number of input channels (e.g. mel bins / out_dims)
    - n_layers: number of residual layers
    - n_chans: residual (hidden) channels width
    - n_hidden: conditioning channels expected (e.g. encoder embedding dim). If None no cond.
    Notes:
      - Forward signature: out = model(x, t=None, cond=None)
        * x: [B, C_in, T]  (or [B,1,C_in,T] where the 1 is squeezed)
        * t: [B] time steps (unused but kept for compatibility)
        * cond: [B, n_hidden, T]  (time-aligned conditioning) or None
      - Output shape equals input channels: [B, C_in, T]
    """
    def __init__(self, in_channels, n_layers=20, n_chans=384, n_hidden=None,
                 kernel_size=3, skip_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = n_chans
        self.n_layers = n_layers
        self.cond_channels = n_hidden
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.skip_channels = skip_channels if skip_channels is not None else n_chans

        # initial 1x1 conv to map input channels -> residual hidden channels
        self.input_proj = nn.Conv1d(in_channels, self.hidden_channels, kernel_size=1)

        # in_layers: dilated convs producing 2*hidden (filter&gate)
        self.in_layers = nn.ModuleList()
        # res_skip_layers: 1x1 convs mapping gated activation -> residual + skip
        # (we implement res/skip inside ResidualGatedBlock)
        # For conditioning: a single conv to produce per-layer cond slices if cond provided
        if self.cond_channels is not None:
            # project conditioning to a concatenation of per-layer (2*hidden) slices
            self.cond_layer = nn.Conv1d(self.cond_channels, self.n_layers * 2 * self.hidden_channels, kernel_size=1, bias=False)
        else:
            self.cond_layer = None

        # build residual stack using dilation cycle (powers of two)
        # choose cycle length so dilations don't explode; common choice is cycle length = 10 or log2(n_layers)
        cycle = min(10, max(1, int(math.log2(max(2, n_layers))) if n_layers > 1 else 1))
        for i in range(n_layers):
            # dilation cycle: 1,2,4,...,2^(cycle-1), repeat
            dilation = 2 ** (i % cycle)
            block = ResidualGatedBlock(hidden_channels=self.hidden_channels,
                                       kernel_size=self.kernel_size,
                                       dilation=dilation,
                                       cond_channels=self.cond_channels if self.cond_layer is not None else None,
                                       dropout=self.dropout)
            self.in_layers.append(block)

        # final processing of accumulated skips -> map to input channels
        self.post_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, in_channels, kernel_size=1)
        )

    def forward(self, x, t=None, cond=None):
        """
        x: [B, C_in, T]  or [B,1,C_in,T] (squeezed) or [B, C_in, F, T] (4D)
        t: [B] time steps (unused here, kept for compatibility with diffusion call)
        cond: [B, n_hidden, T]  (time-aligned) or None
        returns: predicted noise/data with same shape as input
        """
        original_shape = x.shape
        is_4d = False
        
        # support 4D input [B, C_in, F, T] from some callers
        if x.dim() == 4:
            is_4d = True
            B, C, F, T = x.shape
            # Reshape to [B*F, C, T] for processing
            x = x.permute(0, 2, 1, 3).reshape(B * F, C, T)
        
        # support 4D input [B,1,C_in,T] (squeeze middle dim)
        elif x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        if x.dim() != 3:
            raise ValueError(f"WaveNet expects x shape [B,C,T] or [B,1,C,T] or [B,C,F,T], got {original_shape}")

        B_proc = x.shape[0]
        T_proc = x.shape[2]

        # project input
        h = self.input_proj(x)  # [B_proc, hidden, T_proc]

        # prepare conditioning projection if available
        if self.cond_layer is not None and cond is not None:
            if cond.dim() == 4 and cond.shape[1] == 1:
                cond = cond.squeeze(1)
            # cond expected [B, cond_channels, T]
            cond_proj = self.cond_layer(cond)  # [B, n_layers*2*hidden, T]
        else:
            cond_proj = None

        # accumulation of skip outputs - fix: use B_proc instead of B
        skip_sum = torch.zeros((B_proc, self.hidden_channels, T_proc), device=h.device, dtype=h.dtype)

        for i, block in enumerate(self.in_layers):
            # get per-layer cond slice if available
            if cond_proj is not None:
                start = i * 2 * self.hidden_channels
                end = start + 2 * self.hidden_channels
                cond_slice = cond_proj[:, start:end, :]
            else:
                cond_slice = None

            res, skip = block(h, cond_slice=cond_slice)
            # residual connection: add back to h
            h = h + res
            skip_sum = skip_sum + skip

        out = self.post_net(skip_sum)  # [B_proc, in_channels, T_proc]
        
        # Reshape back to original format if input was 4D
        if is_4d:
            out = out.reshape(B, F, self.in_channels, T).permute(0, 2, 1, 3)  # [B, C, F, T]
        
        return out