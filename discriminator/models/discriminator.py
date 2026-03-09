import numpy as np
import torch
import torch.nn as nn


# class Conv1d(nn.Module):
#     """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""
#
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
#         super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
#                                      kernel_size= kernel_size, stride=stride, dilation=dilation,
#                                      groups=groups, bias=bias, padding=padding)
#
#     def forward(self,x):
#         return super().forward(x.transpose(2,1)).transpose(2,1)

class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def forward(self, x):
        # (B, T, C) -> (B, C, T) -> Conv1d -> (B, T, C)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class SingleWindowDisc(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=3, channel=128):
        super(SingleWindowDisc, self).__init__()
        n_layers = 12
        kernel_size = kernel
        dilation = 1

        padding = (kernel - 1) // 2 * dilation
        use_weight_norm = True
        conv_layers = [
            nn.Sequential(
                Conv1d(freq_length, channel, kernel_size=kernel_size,
                       padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]
        for i in range(n_layers - 2):
            conv_layers += [
                Conv1d(channel, channel, kernel_size=kernel_size,
                       padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        conv_layers += [
            nn.Sequential(
                Conv1d(channel, 1, kernel_size=kernel_size,
                       padding=(kernel - 1) // 2, dilation=1, bias=True),
            )
        ]
        self.conv_layers = nn.ModuleList(conv_layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)


    def forward(self, x):
        """
        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.conv_layers:
            x = l(x)
            h.append(x)
        h.pop() # pop final result
        validity = x.view(x.shape[0], -1)
        return validity, h


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, freq_length=80, kernel=3, channel=128):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.discriminators = nn.ModuleList([
            SingleWindowDisc(t, freq_length, kernel, channel=channel)
            for t in time_lengths
        ])


    def forward(self, x, x_len, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        assert len(start_frames_wins) == len(self.discriminators), ""
        h = []
        for i, start_frames in enumerate(start_frames_wins):
            x_clip, start_frames = self.clip(x, x_len, self.win_lengths[i], start_frames)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            x_clip, h_ = self.discriminators[i](x_clip)
            h += h_
            validity.append(x_clip)
        if len(validity) != len(self.discriminators):
            return None, start_frames_wins, h
        validity = sum(validity)  # [B]
        return validity, start_frames_wins, h

    def clip(self, x, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        start_frame_max = (x_len.max() - win_length).item()
        if start_frame_max < 0:
            return None, start_frames

        if start_frames is None:
            T_starts = [0] * x.shape[0]
            T_ends = [l.item()-win_length for l in x_len]
            T_ends = np.clip(T_ends, 0, start_frame_max+1).tolist()
            start_frames = [
                np.random.randint(low=T_starts[i], high=T_ends[i]+1)
                for i in range(x.shape[0])
            ]
        else:
            start_frames = np.clip(start_frames, 0, start_frame_max).tolist()
        x_batch = torch.cat([x[[i], s:s+win_length] for i,s in enumerate(start_frames)])
        return x_batch, start_frames

class MultibandFrequencyDiscriminator(nn.Module):
    def __init__(self, time_lengths, freq_lengths, kernel=3, n_bins=80, channel=128):
        super(MultibandFrequencyDiscriminator, self).__init__()
        self.n_mel_channels = n_bins
        self.win_lengths = freq_lengths
        self.discriminators = nn.ModuleList([
            MultiWindowDiscriminator(time_lengths, f, kernel, channel=channel)
            for f in freq_lengths
        ])

    def forward(self, x, x_len, start_frames_wins=None):
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        assert len(start_frames_wins) == len(self.discriminators), ""

        start_bands_wins = [
            0, # low frequency band
            (self.n_mel_channels - self.win_lengths[1]) // 2, # middle frequency band
            self.n_mel_channels - self.win_lengths[2] # high frequency band
        ]
        validity = []
        start_frames = []
        h = []
        for i, start_bands in enumerate(start_bands_wins):
            freqband = x[..., start_bands : start_bands + self.win_lengths[i]]
            validity_, start_frames_wins_, h_ = self.discriminators[i](freqband, x_len, start_frames_wins[i])
            h.append(h_)
            start_frames.append(start_frames_wins_)
            validity.append(validity_)

        if len(validity) != len(self.discriminators):
            return  None, start_frames_wins, h
        validity = sum(validity) # [B, 1]
        return validity, start_frames, h



class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        time_lengths = hparams.time_lengths
        freq_lengths = hparams.freq_lengths
        kernel = hparams.kernel_size
        channel = hparams.hidden_size

        self.time_lengths = time_lengths
        self.discriminator = MultibandFrequencyDiscriminator(
            freq_lengths=freq_lengths,
            time_lengths=time_lengths,
            kernel=kernel,
            channel=channel
        )


    def forward(self, x, x_len=None, start_frames_wins=None):
        """

        :param x: [B, T, Mel]
        :param x_len: [B,]:
        :return:
        """
        if x_len is None:
            x_len = x.sum([-1]).ne(0).int().sum([-1]) # [B,]
        y, start_frames_wins, h = self.discriminator(
            x, x_len, start_frames_wins=start_frames_wins
        )
        y = y.squeeze(-1)
        return y, start_frames_wins, h

if __name__ == '__main__':
    inputs = torch.randn(2, 600, 80)
    import sys; sys.path.insert(0, '..')
    from utils.hparams import HParams
    hparams = HParams('../config/default.yaml')
    net = Discriminator(hparams.discriminator)
    print(net)
    y, start_frames_wins, h = net(inputs, None)
    print(y.shape)
    print(start_frames_wins)
    h_list = [h_in for h_out in h for h_in in h_out]
    print(torch.stack(h_list, dim=-2).shape)
    y, start_frames_wins, h = net(inputs, None, start_frames_wins=start_frames_wins)
    print(start_frames_wins)
