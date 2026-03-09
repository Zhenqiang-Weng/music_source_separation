""" CC model class """
import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator.utils.functional import get_mask_from_lengths
from .layers import TextEncoder, SpecDecoder, RhythmAdaptor, Interpolate
from .losses import masked_mse, masked_mae, masked_huber,masked_ssim, masked_nll, dtw, GuidedAttentionLoss

class CC(nn.Module):
    def __init__(self, hparams):
        """Text to Melspectrogram Model

        :param hparams:
        """
        super(CC, self).__init__()
        self.hparams = hparams
        self.encoder = TextEncoder(hparams)
        self.decoder = SpecDecoder(hparams)
        self.adaptor = RhythmAdaptor(hparams)

    def parse_batch(self, batch, device=None):
        text_inputs = batch['texts'].to(device)
        breaks = batch['brks'].to(device)
        input_lengths = batch['lens'].to(device)
        specs = batch['specs'].to(device)
        word_boundaries = batch['word_boundaries'].to(device)
        src_w_lens = batch['text_w_lens'].to(device)
        attn_priors = batch['attn'].to(device)
        durations = batch['drns'].to(device)
        speakers = batch['spks'].to(device) if self.hparams.multi_speakers else None
        styles = batch['stls'].to(device) if self.hparams.multi_styles else None
        pitches = batch['f0s'].to(device) if self.hparams.enable_pitch else None
        energies = batch['nrgs'].to(device) if self.hparams.enable_energy else None

        duration = (durations, breaks)
        features = (pitches, energies)
        voice = (speakers, styles)
        controls = (None, None, None)
        inputs = (text_inputs, input_lengths, duration, features, voice, controls, word_boundaries, src_w_lens, attn_priors)
        targets = (specs, durations, features)
        output_lengths = torch.sum(durations, dim=-1)
        lengths = (input_lengths, output_lengths)

        return (inputs, targets, lengths)

    def parse_infer_batch(self, batch, controls=None, device=None):
        text_inputs = batch['texts'].to(device)
        breaks = batch['brks'].to(device)
        input_lengths = batch['lens'].to(device)
        word_boundaries = batch['word_boundaries'].to(device)
        src_w_lens = batch['text_w_lens'].to(device)
        # attn_priors = batch['attn'].to(device)

        speakers = batch['spks'].to(device) if self.hparams.multi_speakers else None
        styles = batch['stls'].to(device) if self.hparams.multi_styles else None

        duration = (None, breaks)
        features = (None, None) # pitches, energies
        voice = (speakers, styles)
        controls = controls or (1.0, 0.0, 0.0)
        inputs = (text_inputs, input_lengths, duration, features, voice, controls, word_boundaries, src_w_lens, None)

        return  inputs

    def forward(self, inputs):
        text_inputs, input_lengths, duration, features, voice, controls, word_boundaries, src_w_lens, attn_priors = inputs
        output_lengths = torch.sum(duration[0], dim=-1) # if duration[0] else None
        max_mel_lens = max(output_lengths) #if output_lengths else None
        mel_masks = (
            get_mask_from_lengths(output_lengths, max_mel_lens)
            if output_lengths is not None else None
        )
        src_masks = get_mask_from_lengths(input_lengths, max(input_lengths))

        src_w_masks = get_mask_from_lengths(src_w_lens, max(src_w_lens))

        encodings = self.encoder(text_inputs, input_lengths) # [B, N, DIM]

        (
            encodings, output_lengths, alignment_outputs, feature_outputs
        ) = self.adaptor(
            encodings, duration, features, voice, input_lengths, src_masks, word_boundaries,\
            src_w_lens, src_w_masks, max_mel_lens, controls, mel_masks, attn_priors
        )
        # [B, T, DIM], [B,], [B, N] [B, N, T]

        spec_outputs = self.decoder(encodings, output_lengths) # [B, T, Mel]

        return spec_outputs, output_lengths, alignment_outputs, feature_outputs, src_w_lens

class CCLoss(nn.Module):
    def __init__(self, hparams):
        super(CCLoss, self).__init__()
        self.hparams = hparams
        self.counter = 0
        self.guided_attn_loss = GuidedAttentionLoss(
            sigma=0.4,
            alpha=1.0,
        )

    def forward(self, outputs_and_targets, use_dtw=False):
        outputs, targets, lengths = outputs_and_targets
        spec_outputs, predicted_lengths, alignment_outputs, feature_outputs, src_w_lens = outputs
        # duration_outputs, route_weights, alignments, hiddens = alignment_outputs
        log_duration_w_prediction, duration_w_rounded, route_weights, attns, attn_logprob, hiddens = alignment_outputs
        pitch_outputs, energy_outputs = feature_outputs
        specs, durations, features = targets
        spd_target = durations.float()
        pitches, energies = features
        input_lengths, output_lengths = lengths


        if use_dtw:
            mse_loss = dtw(spec_outputs, predicted_lengths, specs, output_lengths, pow=2.0)
            mae_loss = dtw(spec_outputs, predicted_lengths, specs, output_lengths, pow=1.0)
        else:
            mse_loss = masked_mse(spec_outputs, specs, output_lengths)
            mae_loss = masked_mae(spec_outputs, specs, output_lengths)

        losses = {'mae': mae_loss, 'mse': mse_loss}


        if self.hparams.enable_ssim_loss:
            ssim_loss = torch.tensor(0.0) if use_dtw else \
                sum([masked_ssim(spec_outputs, specs, output_lengths, window_size=w) for w in self.hparams.ssim_window_size])
            losses['ssim'] = ssim_loss

        self._update_loss_lambda() # update partial loss lambda

        if self.hparams.duration_quantization =='log':
            log_duration_targets = torch.log(duration_w_rounded.float() + 1.0)
        drn_loss = masked_huber(log_duration_w_prediction, log_duration_targets, src_w_lens)
        losses['drn'] = drn_loss * self.hparams.duration_loss_lambda

        attn_loss = 0
        for alignment in attns[1]:
            attn_loss += self.guided_attn_loss(alignment, input_lengths, output_lengths)
        losses['attention'] = attn_loss

        if self.hparams.enable_cosine_loss:
            # cosine distance loss
            cos_specs = F.cosine_similarity(specs[:, :-1, :], specs[:, 1:, :], dim=-1)
            cos_hiddens = F.cosine_similarity(hiddens[:, :-1, :], hiddens[:, 1:, :], dim=-1)
            cos_loss = masked_huber(cos_specs, cos_hiddens, output_lengths-1)
            losses['cos'] = cos_loss * (1 - self.hparams.duration_loss_lambda) * 10

        if self.hparams.enable_speed_moe:
            lower, upper = self.hparams.speed_boundaries[0], self.hparams.speed_boundaries[-1]
            boundaries = torch.Tensor(self.hparams.speed_boundaries[1:-1]).to(spd_target.device)
            boundaries.requires_grad = False
            spd_target = torch.bucketize(spd_target.clamp(lower, upper), boundaries)
            spd_target.requires_grad = False
            route_weights = torch.log(route_weights).permute(0,2,1) # [B, N_exp, L]
            spd_loss = masked_nll(route_weights, spd_target, input_lengths)
            losses['spd'] = spd_loss * self.hparams.duration_loss_lambda


        if self.hparams.enable_pitch:
            f0_loss = masked_huber(pitch_outputs, pitches, input_lengths)
            losses['f0'] = f0_loss * self.hparams.features_loss_lambda * 0.2

        if self.hparams.enable_energy:
            nrg_loss = masked_huber(energy_outputs, energies, input_lengths)
            losses['nrg'] = nrg_loss * self.hparams.features_loss_lambda

        self.counter += 1

        return losses

    def _update_loss_lambda(self):
        self.hparams.duration_loss_lambda = (max(40, self.counter) * 0.025) ** (-0.5)




