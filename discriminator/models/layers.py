import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator.utils.functional import get_mask_from_lengths
from .modules import Conv1d, FreqNorm, Mask
from .constants import ACTIVATIONS, NORMALIZATIONS
from .blocks import ResidualBlock, RelativeFFTBlock, word_level_pooling, WordToPhonemeAttention
from .sublayers import VariancePredictor, VanillaUpsampler, GaussianUpsampler, PositionalEncoding

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class TextEncoder(nn.Module):
    """Encodes input for the duration predictor and the decoder"""
    def __init__(self, hp):
        super(TextEncoder, self).__init__()
        self.kernel_size = hp.enc_kernel_size
        self.dilations = hp.enc_dilations

        self.prenet = nn.Sequential(
            nn.Embedding(hp.n_symbols, hp.channels, padding_idx=0),
            Conv1d(hp.channels, hp.channels),
            ACTIVATIONS[hp.activation](),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hp.channels, hp.kernel_size, d, n=2, norm=hp.normalize, activation=hp.activation)
            for d in self.dilations
        ])

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            ACTIVATIONS[hp.activation](),
            NORMALIZATIONS[hp.normalize](hp.channels),
            Conv1d(hp.channels, hp.channels)
        )

    def forward(self, text_inputs, input_lengths):

        embedding = self.prenet(text_inputs)
        encodings = self.res_blocks(embedding)
        encodings = self.post_net1(encodings) + embedding
        encodings = self.post_net2(encodings)
        encodings = Mask(0)(encodings, input_lengths, dim=1)
        return encodings

class SpecDecoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""
    def __init__(self, hp):
        super(SpecDecoder, self).__init__()
        self.kernel_size = hp.dec_kernel_size
        self.dilations = hp.dec_dilations

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hp.channels, self.kernel_size, d, n=2, norm=hp.normalize, activation=hp.activation)
            for d in self.dilations],
        )

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            ResidualBlock(hp.channels, self.kernel_size, 1, n=2, norm=hp.normalize, activation=hp.activation),
            Conv1d(hp.channels, hp.out_channels),
            ACTIVATIONS[hp.final_activation]()
        )

    def forward(self, encodings, output_lengths):
        spec_outputs = self.res_blocks(encodings)

        spec_outputs = self.post_net1(spec_outputs) + encodings
        spec_outputs = self.post_net2(spec_outputs)
        spec_outputs = Mask(0)(spec_outputs, output_lengths, dim=1)

        return spec_outputs

class MoEPredictor(nn.Module):
    def __init__(self, channels, out_channels=1, experts=3, kernel=3, norm='freq', dropout=0.1):
        super(MoEPredictor, self).__init__()

        self.router = VariancePredictor(channels, out_channels=experts, kernel=kernel, norm=norm, dropout=dropout)
        self.experts = nn.ModuleList([
            VariancePredictor(channels, out_channels, kernel=kernel, norm=norm, dropout=dropout) for i in range(experts)
        ])

    def forward(self, x):
        weights = F.softmax(self.router(x), dim=-1) # [B, L, N_exp]

        outputs = [e(x) for e in self.experts] # N_exp *  [B, L, C]
        outputs = torch.stack(outputs) # [N_exp, B, L, C]

        outputs = weights.permute(2,0,1).unsqueeze(-1) * outputs #[N_exp, B, L, C]
        outputs = torch.sum(outputs, dim=0) # [B, L, C]

        return outputs, weights

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class RhythmAdaptor(nn.Module):
    """Rhythm Adaptor"""
    QUANTIZATIONS = ['linear', 'log']

    def __init__(self, hp):
        super(RhythmAdaptor, self).__init__()
        self.hparams = hp
        self.max_seq_len = hp.max_seq_len
        # word encoder
        self.word_encoder = RelativeFFTBlock(
            hidden_channels=hp.channels,
            filter_channels=hp.conv_filter_size,
            n_heads=hp.encoder_head,
            n_layers=hp.encoder_layer,
            kernel_size=hp.conv_kernel_size,
            window_size=hp.encoder_window_size,
        )
        self.kv_position_enc = nn.Parameter(
            get_sinusoid_encoding_table(hp.max_seq_len + 1, hp.channels).unsqueeze(0),
            requires_grad=True,
        )
        self.q_position_enc = nn.Parameter(
            get_sinusoid_encoding_table(hp.max_seq_len + 1, hp.channels).unsqueeze(0),
            requires_grad=True,
        )
        self.w2p_attn = WordToPhonemeAttention(
            hp.encoder_head, hp.channels, hp.channels // hp.encoder_head, hp.channels // hp.encoder_head
        )

        # Durations
        assert hp.duration_quantization in self.QUANTIZATIONS
        if hp.enable_speed_moe:
            self.duration_predictor = MoEPredictor(hp.channels,
                                                   experts=len(hp.speed_boundaries)-1, kernel=hp.enc_kernel_size,
                                                   norm=hp.normalization, dropout=hp.dropout)
        else:
            self.duration_predictor = VariancePredictor(hp.channels, kernel=hp.enc_kernel_size,
                                                   norm=hp.normalization, dropout=hp.dropout)
        # if hp.use_gaussian_upsample:
        #     self.range_predictor = VariancePredictor(hp.channels+1, kernel=hp.enc_kernel_size,
        #                                            norm=hp.normalization, dropout=hp.dropout)
        #     self.gaussian_upsampler = GaussianUpsampler(hp.pos_mode)
        # else:
        #     self.vanilla_upsampler = VanillaUpsampler(hp.pos_mode)
        self.length_regulator = LengthRegulator()
        self.pos_encoding = PositionalEncoding(hp.channels)

        # Features
        if hp.enable_break:
            self.brs_embedding = nn.Embedding(hp.n_breaks, hp.channels)

        if hp.enable_pitch:
            self.pitch_bins = nn.Parameter(
                torch.linspace(hp.pitch_range[0], hp.pitch_range[1], hp.pitch_bins -1),
                requires_grad=False
            )
            self.pitch_embedding = nn.Embedding(hp.pitch_bins, hp.channels)
            self.pitch_predictor = VariancePredictor(hp.channels, kernel=hp.enc_kernel_size,
                                                   norm=hp.normalization, dropout=hp.dropout)

        if hp.enable_energy:
            self.energy_bins = nn.Parameter(
                torch.linspace(hp.energy_range[0], hp.energy_range[1], hp.energy_bins -1),
                requires_grad=False
            )
            self.energy_embedding = nn.Embedding(hp.energy_bins, hp.channels)
            self.energy_predictor = VariancePredictor(hp.channels, kernel=hp.enc_kernel_size,
                                                   norm=hp.normalization, dropout=hp.dropout)
        # Embeddings
        if hp.multi_speakers:
            self.spk_embedding = nn.Embedding(hp.n_speakers, hp.channels)
        if hp.multi_styles:
            self.stl_embedding = nn.Embedding(hp.n_styles, hp.channels)
        # Interpolate
        if hp.interpolate:
            self.interpolate = Interpolate(hp)

    def get_durations(self, durations, lengths, control):
        if self.hparams.duration_quantization == 'log':
            durations = torch.exp(durations) - 1.0

        durations *= (1 / control)
        # ensure the length of b/e silence
        indices = torch.arange(durations.size(0))
        durations[indices, 0] = self.hparams.max_frames_per_phoneme
        durations[indices, lengths-1] = self.hparams.max_frames_per_phoneme
        durations = torch.clamp(torch.round(durations),
                                min=self.hparams.min_frames_per_phoneme,
                                max=self.hparams.max_frames_per_phoneme).long()
        durations = Mask(0)(durations, lengths, dim=1)

        return durations

    def get_pitch_embedding(self, encodings, pitches, lengths, control):
        pitch_outputs = self.pitch_predictor(encodings).squeeze(-1)
        if pitches is not None:
            pitches = pitch_outputs + control
            pitches = torch.clamp(pitches, *self.hparams.pitch_range)
            pitches = Mask(0)(pitches, lengths, dim=1)
        embedding = self.pitch_embedding(torch.bucketize(pitches, self.pitch_bins))

        return pitch_outputs, embedding

    def get_energy_embedding(self, encodings, energies, lengths, control):
        energy_outputs = self.energy_predictor(encodings).squeeze(-1)
        if energies is not None:
            energies = energy_outputs + control
            energies = torch.clamp(energies, *self.hparams.energy_range)
            energies = Mask(0)(energies, lengths, dim=1)
        embedding = self.energy_embedding(torch.bucketize(energies, self.energy_bins))

        return energy_outputs, embedding

    def add_position_enc(self, src_seq, position_enc=None, coef=None):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            pos_enc = get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        else:
            position_enc = self.abs_position_enc if position_enc is None else position_enc
            pos_enc = position_enc[
                      :, :max_len, :
                      ].expand(batch_size, -1, -1)
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        return enc_output

    def get_mapping_mask(self, q, kv, dur_w, wb, src_w_len):
        """
        For applying a word-to-phoneme mapping mask to the attention weight to force each query (Q)
        to only attend to the phonemes belongs to the word corresponding to this query.
        """
        batch_size, q_len, kv_len, device = q.shape[0], q.shape[1], kv.shape[1], kv.device
        mask = torch.ones(batch_size, q_len, kv_len, device=device)
        for b, (w, p, l) in enumerate(zip(dur_w, wb, src_w_len)):
            w, p = [0] + [d.item() for d in torch.cumsum(w[:l], dim=0)], [0] + \
                   [d.item() for d in torch.cumsum(p[:l], dim=0)]
            # assert len(w) == len(p)
            for i in range(1, len(w)):
                mask[b, w[i - 1]:w[i], p[i - 1]:p[i]
                ] = torch.zeros(w[i] - w[i - 1], p[i] - p[i - 1], device=device)
        return mask == 0.

    def add_position_enc(self, src_seq, position_enc=None, coef=None):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            pos_enc = get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        else:
            position_enc = self.abs_position_enc if position_enc is None else position_enc
            pos_enc = position_enc[
                      :, :max_len, :
                      ].expand(batch_size, -1, -1)
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        return enc_output

    def get_rel_coef(self, dur, dur_len, mask):
        """
        For adding a well-designed positional encoding to the inputs of word-to-phoneme attention module.
        """
        idx, L, device = [], [], dur.device
        for d, dl in zip(dur, dur_len):
            idx_b, d = [], d[:dl].long()
            m = torch.repeat_interleave(d, torch.tensor(
                list(d), device=device), dim=0)  # [tgt_len]
            L.append(m)
            for d_i in d:
                idx_b += list(range(d_i))
            idx.append(torch.tensor(idx_b).to(device))
            # assert L[-1].shape == idx[-1].shape
        return torch.div(pad(idx).to(device), pad(L).masked_fill(mask == 0., 1.).to(device))

    def forward(self,encodings,duration, features, voice, input_lengths, src_masks,
                word_boundaries, src_w_lens, src_w_masks, max_mel_lens, controls, mel_masks=None, attn_priors=None):
        durations, breaks = duration # [B, N], [B, N]
        pitches, energies = features # [B, N or T], [B, N or T]
        speakers, styles = voice # [B,], [B,]
        control, p_control, e_control = controls

        pitch_outputs = energy_outputs = None

        # Voices
        if self.hparams.multi_speakers:
            speaker_embeddings = self.spk_embedding(speakers)
            encodings += speaker_embeddings.unsqueeze(1) # [B, N, DIM]
        if self.hparams.multi_styles:
            style_embeddings = self.stl_embedding(styles)
            encodings += style_embeddings.unsqueeze(1) # [B, N, DIM]

        if self.hparams.enable_break:
            encodings += self.brs_embedding(breaks)


        if self.hparams.feature_level == 'phoneme':
            if self.hparams.enable_pitch: # Pitch
                pitch_outputs, pitch_embedding = self.get_pitch_embedding(
                    encodings, pitches, input_lengths, control=p_control
                )
                encodings += pitch_embedding

            if self.hparams.enable_energy: # Energy
                energy_outputs, energy_embedding = self.get_energy_embedding(
                    encodings, energies, input_lengths, control=e_control
                )
                encodings += energy_embedding
        # word-level pooling
        src_w_seq = word_level_pooling(
            encodings, input_lengths, word_boundaries, src_w_lens, reduce='mean'
        )
        # word encoding
        enc_w_out = self.word_encoder(
            src_w_seq.transpose(1,2),
            src_w_masks.unsqueeze(1)
        ).transpose(1,2)


        duration_inputs = encodings.detach() \
            if self.hparams.separate_duration_grad else encodings # [B, N, DIM]
        if self.hparams.enable_speed_moe:
            duration_outputs, route_weights = self.duration_predictor(duration_inputs) # [B, N, 1], [B, N, N_exp]
        else:
            duration_outputs, route_weights = self.duration_predictor(duration_inputs), None
        duration_outputs = duration_outputs.squeeze(-1) # [B, N]

        log_duration_w_prediction = torch.log(word_level_pooling(
            duration_outputs.exp().unsqueeze(-1), input_lengths, word_boundaries,src_w_lens,reduce='sum'
        ) + 1).squeeze(-1)
        x = enc_w_out


        # Inference Only
        if durations is None:
            duration_w_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_w_prediction) - 1) * control),
                min=0,
            ).long()
            x, mel_len = self.length_regulator(x, duration_w_rounded, max_mel_lens)
            mel_masks = get_mask_from_lengths(mel_len)
        else:
            duration_w_rounded = word_level_pooling(
                durations.unsqueeze(1), input_lengths, word_boundaries, src_w_lens,reduce="sum"
            ).squeeze(-1)
            x, mel_len = self.length_regulator(x, duration_w_rounded, max_mel_lens)


        # word-to-phoneme attention
        # [batch, mel_len, seq_len]
        src_mask_ = src_masks.unsqueeze(1).expand(-1, mel_masks.shape[1], -1)
        # [batch, mel_len, seq_len]
        mel_mask_ = mel_masks.unsqueeze(1).expand(-1, -1, src_masks.shape[1])
        # [batch, mel_len, seq_len]
        mapping_mask = self.get_mapping_mask(
            x, encodings, duration_w_rounded, word_boundaries, src_w_lens
        )

        q = self.add_position_enc(x, position_enc=self.q_position_enc, coef=self.get_rel_coef(
            duration_w_rounded,src_w_lens, mel_masks
        ))
        k = self.add_position_enc(encodings, position_enc=self.kv_position_enc, coef=self.get_rel_coef(
            word_boundaries,input_lengths, src_masks
        ))
        v = self.add_position_enc(encodings, position_enc=self.kv_position_enc, coef=self.get_rel_coef(
            word_boundaries, input_lengths, src_masks
        ))

        x, attns, attn_logprob = self.w2p_attn(
            q=q,
            k=k,
            v=v,
            key_mask=src_mask_,
            query_mask=mel_mask_,
            mapping_mask=mapping_mask,
            indivisual_attn=True,
            attn_prior=None,
        )


        output_lengths = torch.sum(duration_w_rounded, dim=-1) # [B,]


        encodings = Mask(0)(x, output_lengths, dim=1) # [B, T, DIM]

        if self.hparams.feature_level == 'frame':
            if self.hparams.enable_pitch:  # Pitch
                pitch_outputs, pitch_embedding = self.get_pitch_embedding(
                    encodings, pitches, output_lengths, control=p_control
                )
                encodings += pitch_embedding

            if self.hparams.enable_energy:  # Energy
                energy_outputs, energy_embedding = self.get_energy_embedding(
                    encodings, energies, output_lengths, control=e_control
                )
                encodings += energy_embedding


        if  self.hparams.interpolate:
            encodings = self.interpolate(encodings)

            if self.hparams.multi_speakers:
                encodings += speaker_embeddings.unsqueeze(1) # [B, T, DIM]
            if self.hparams.multi_styles:
                encodings += style_embeddings.unsqueeze(1) # [B, T, DIM]
        hiddens = encodings.clone() # hidden encodings
        alignment_outputs = (log_duration_w_prediction, duration_w_rounded, route_weights, attns, attn_logprob, hiddens)
        feature_outputs = (pitch_outputs, energy_outputs)

        return encodings, output_lengths, alignment_outputs, feature_outputs



class Interpolate(nn.Module):
    """Use multihead attention to increase variability in expanded phoneme encodings

    Not used in the final model, but used in reported experiments.
    """

    def __init__(self, hp):
        super(Interpolate, self).__init__()

        self.att = nn.MultiheadAttention(hp.channels, num_heads=4)
        self.norm = NORMALIZATIONS[hp.normalization](hp.channels)
        self.conv = Conv1d(hp.channels, hp.channels, kernel_size=1)

    def forward(self, x):
        xx = x.permute(1, 0, 2)  # [B, T, DIM] -> [T, B, DIM]
        xx = self.att(xx, xx, xx)[0].permute(1, 0, 2)  # [B, T, DIM]
        xx = self.conv(xx)
        return self.norm(xx) + x