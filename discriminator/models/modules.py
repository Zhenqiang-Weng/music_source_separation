import torch.nn as nn
import torch



class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


class Conv(nn.Module):
    """Conv1d/CausalConv1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, causal=False):
        """Initialize Conv1d/CausalConv1d module."""
        super(Conv, self).__init__()
        self.pad = ZeroTemporalPad(kernel_size, dilation, causal=causal)
        self.causal = causal
        self.conv = Conv1d(in_channels, out_channels, kernel_size,
                           stride=1,  # paper: 'The stride of convolution is always 1.'
                           dilation=dilation, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.

        :param x: (Tensor) Input tensor (batch, time, in_channels).
        :returns: (Tensor) Output tensor (batch, time, out_channels).
        """
        return self.conv(self.pad(x))


class FreqNorm(nn.BatchNorm1d):
    """Normalize separately each frequency channel in spectrogram and batch,


    Examples:
        t = torch.arange(2*10*5).reshape(2, 10, 5).float()
        b1 = nn.BatchNorm1d(10, affine=False, momentum=None)
        b2 = (t - t.mean([0,2], keepdim=True))/torch.sqrt(t.var([0,2], unbiased=False, keepdim=True)+1e-05)
        -> b1 and b2 give the same results
        -> BatchNorm1D by default normalizes over channels and batch - not useful for differet length sequences
        If we transpose last two dims, we get normalizaton across batch and time
        -> normalization for each frequency channel over time and batch

        # compare to layer norm:
        Layer_norm: (t - t.mean(-1, keepdim=True))/torch.sqrt(t.var(-1, unbiased=False, keepdim=True)+1e-05)
        -> layer norm normalizes across all frequencies for each timestep independently of batch

        => LayerNorm: Normalize each freq. bin wrt to other freq bins in the same timestep -> time independent, batch independent, freq deendent
        => FreqNorm: Normalize each freq. bin wrt to the same freq bin across time and batch -> time dependent, other freq independent
    """

    def __init__(self, channels, affine=True, track_running_stats=True, momentum=0.1):
        super(FreqNorm, self).__init__(channels, affine=affine, track_running_stats=track_running_stats,
                                       momentum=momentum)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


class Pad(nn.ZeroPad2d):
    def __init__(self, kernel_size, dilation):
        pad_total = dilation * (kernel_size - 1)
        begin = pad_total // 2
        end = pad_total - begin

        super(Pad, self).__init__((begin, end, begin, end))


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""

    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))

        if causal:
            super(ZeroTemporalPad, self).__init__((0, 0, total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((0, 0, begin, end))

class Mask(nn.Module):
    def __init__(self, mask_value=0):
        super(Mask, self).__init__()
        self.mask_value = mask_value

    def forward(self, x, lengths, dim=-1):
        assert dim != 0, 'Masking not available for batch dimension'
        assert len(lengths) == x.shape[0], 'Lengths must contain as many elements as ther are items in the batch'

        lengths = torch.as_tensor(lengths)

        to_expand = [1] * (len(x.shape)-1) + [-1]
        mask = torch.arange(x.shape[dim]).expand(to_expand).transpose(dim, -1).expand(x.shape).to(lengths.device)
        mask = mask < lengths.expand(to_expand).transpose(0, -1)

        masked_x = x.masked_fill(~mask, self.mask_value)
        return masked_x