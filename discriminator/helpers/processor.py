__author__ = 'Atomicoo'

import os
import math
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import librosa
import pyworld as pw
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.functional import  get_duration_from_file, interpolate, aggregate_by_duration
from utils.util import load_wavefile



class TextProcessor:
    def __init__(self, hparams):
        self.hparams = hparams

        self.symbols = hparams.symbols
        self.specials = hparams.specials
        self.punctuations = hparams.punctuations
        self.symbols = self.specials + self.symbols + self.punctuations

        # symbols mapping
        self.sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.idx2sym = {idx: sym for idx, sym in enumerate(self.symbols)}

        # break labels:
        # br0 - 1, br1 - 2, br2 - 3, br3 - 4, br4 - 5
        # sp - n, sil - 6
        self.break_labels = ['br0', 'br1', 'br2', 'br3', 'br4']
        # stress labels
        self.stress_labels = ['1', '2']

    def text_to_breaks(self, text):
        breaks = []
        for i in range(len(text)):
            if text[i] in self.break_labels:
                continue
            if text[i] in self.stress_labels:
                continue
            if text[i-1] == 'sil' or text[i] == 'sil' or text[i+1] == 'sil':
                breaks.append(6)
            elif text[i-1].endswith('sp'):
                breaks.append(int(text[i-1][2])+1)
            elif text[i].endswith('sp'):
                breaks.append(int(text[i][2])+1)
            elif text[i+1].endswith('sp'):
                breaks.append(int(text[i+1][2])+1)
            elif text[i-1] in self.break_labels:
                breaks.append(int(text[i-1][2])+1)
            elif text[i+1] in self.break_labels:
                breaks.append(int(text[i+1][2])+1)
            else:
                breaks.append(0)
        return breaks

    def text_to_sequence(self, text):
        sequence = []
        for tt in text:
            if tt in self.break_labels:
                continue
            if tt in self.stress_labels:
                continue
            if tt.endswith('sp'):
                tt = 'sp'
            sequence.append(self.sym2idx[tt])
        return sequence

    def get_phone_per_word(self, text):
        phones_per_word = []
        sil_phones = ["sil", "sp"]
        phone_num = 0
        num_list = ['0', '1', '2', '3', '4']
        for j in range(len(text)):
            if 'br' in text[j] and len(text[j]) == 3:
                continue
            elif text[j] in num_list:
                continue
            else:
                phone_num += 1
        k = 0
        while k < len(text):
            if text[k] in sil_phones or 'sp' in text[k] or 'sil' in text[k]:
                phones_per_word.append(1)
            elif 'br' in text[k] and len(text[k]) == 3:
                k += 1
            elif k + 2 < len(text) and text[k+2] in num_list:
                k += 3
                phones_per_word.append(2)
            else:
                k += 3
                phones_per_word.append(3)
        return phones_per_word

    def __call__(self, texts, split=' ', min_length=13):
        assert isinstance(texts, (str, list, tuple)), "Inputs must be str or list(str)"
        if isinstance(texts,str):
            texts = [texts]
        assert isinstance(texts[0], str), "Inputs must be str or list(str)"

        texts = [s.strip().split(split)for s in texts]
        text_inputs = [self.text_to_sequence(text) for text in texts]
        breaks = [self.text_to_breaks(text) for text in texts]
        ph_per_words = [self.get_phone_per_word(text) for text in texts]
        input_lengths = [[len(s)] for s in text_inputs]

        if min_length > 0:
            for index in range(len(texts)):
                if input_lengths[index][0] < min_length:
                    # pad sequence to min-length
                    pad_length = min_length - input_lengths[index][0]
                    text_inputs[index] += [0] * pad_length
                    breaks[index] += [0] * pad_length

        return text_inputs, breaks, input_lengths, # ph_per_words


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def normalize_volume(wave, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")

    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wave ** 2))
    if (dBFS_change < 0 and not increase_only) or (dBFS_change > 0 and not decrease_only):
        wave = wave * (10 ** (dBFS_change / 20))
    return wave

def trim_long_silences(wave, sampling_rate, bit_depth=16, window_length=30, moving_average_width=8, max_silence_length=6):
    # Compute the voice detection window size
    samples_per_window = (window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wave = wave[:len(wave) - (len(wave) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    max_wav_value = float(2 ** (bit_depth - 1))
    pcm_wave = struct.pack("%dh" % len(wave), *(np.round(wave * max_wav_value)).astype(np.dtype(f'int{bit_depth}')))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wave), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wave[audio_mask == True], audio_mask



def get_statistics(filelist, nonzero=False):
    sdscaler, mmscaler = StandardScaler(), MinMaxScaler()

    for file in tqdm(filelist):
        try:
            assert os.path.isfile(file), f"File {file} not exists"
            feature = np.load(file)
            if nonzero:
                feature = feature[feature != 0] # indices = np.where(feature != 0)[0]
            # feature = remove_outlier(feature) # remove outliers

            sdscaler.partial_fit(feature.reshape((-1, 1)))
            mmscaler.partial_fit(feature.reshape((-1, 1)))
        except Exception as err:
            print(err); continue

    return [float(mmscaler.data_min_[0]), float(mmscaler.data_max_[0]),
            float(sdscaler.mean_[0]), float(sdscaler.scale_[0])]


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


class MySTFT:
    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_basis = librosa.filters.mel(
            sr=self.hparams.sampling_rate, n_fft=self.hparams.filter_length,
            n_mels=self.hparams.n_mel_channels,
            fmin=self.hparams.mel_fmin, fmax=self.hparams.mel_fmax)


    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def wave_to_melspec(self,wave):
        stft_matrix = librosa.stft(
            wave, n_fft=self.hparams.filter_length,
            hop_length=self.hparams.hop_length, win_length=self.hparams.win_length,
            window=self.hparams.window
        )
        magnitudes, phase = librosa.magphase(stft_matrix, power=self.hparams.power)

        melspec = np.einsum("...ft,mf->...mt", magnitudes, self.mel_basis, optimize=True)
        melspec = self.spectral_normalize(melspec)

        return melspec, magnitudes

    def melspec_to_wave(self, melspec, griffin_iters=60):
        melspec = self.spectral_de_normalize(melspec)
        inverse = librosa.util.nnls(self.mel_basis, melspec)
        magnitudes = np.power(inverse, 1.0 / self.hparams.power, out=inverse)
        wave = self.griffin_lim(magnitudes, n_iter=griffin_iters)

        return wave

    def griffin_lim(self, magnitudes, n_iter=32, momentum=0.99):
        # randomly initialize the phase
        angles = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)).astype(np.complex64)
        eps = librosa.util.tiny(angles)

        # And initialize the previous iterate to 0
        rebuilt = 0.0
        for _ in range(n_iter):
            # Store the previous iterate
            tprev = rebuilt
            # Invert with our current estimate of the phases
            inverse = librosa.istft(
                magnitudes * angles,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                window=self.hparams.window
            )
            # Rebuild the spectrogram
            rebuilt = librosa.stft(
                inverse,
                n_fft=self.hparams.filter_length,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                window=self.hparams.window
            )
            # Update our phase estimates
            angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
            angles[:] /= np.abs(angles) + eps

        return librosa.istft(
                magnitudes * angles,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                window=self.hparams.window
            )

class SpecProcessor:
    def __init__(self, hparams):
        super(SpecProcessor, self).__init__()
        self.hparams = hparams

        self.stft = MySTFT(hparams)

    def __call__(self, filelist):
        """
        Process the given wavefiles.
        :param filelist:
        :return:
        """
        if self.hparams.load_data_from_disk:
            paths, stems = list(), list()

            for file in tqdm(filelist):
                try:
                    path, stem = self.checkfile(file,mode='mel')
                except Exception as err:
                    path, stem = False, False
                    print(err, file)
                finally:
                    paths.append(path)
                    stems.append(stem)

        else:
            paths, stems = list(), list()

            # Generating melspecs
            print('Generating mel-spectrogram ...')


            for file in tqdm(filelist):
                try:
                    path, stem = self.checkfile(file, mode='wav')
                    done = self.preprocess(path,stem,prep=self.hparams.preprocess)
                except Exception as err:
                    path, stem = False, False
                    print(err, file)
                finally:
                    paths.append(path)
                    stems.append(stem)

            print("Finish generate melspecs npy!!")


        if self.hparams.compute_statistics:
            melspec_files = list()
            pitch_files, energy_files = list(), list()
            for path, stem in zip(paths,stems):
                if path is False or stem is False: continue
                melspec_files.append(os.path.join(path, 'mel', f"{stem}.npy"))
                pitch_files.append(os.path.join(path, 'pitch', f"{stem}.npy"))
                energy_files.append(os.path.join(path, 'energy', f"{stem}.npy"))

            melspec_stats = get_statistics(melspec_files)
            print("Finish melspec statistics")
            pitch_stats = get_statistics(pitch_files)
            print("Finish pitch statistics")
            energy_stats = get_statistics(energy_files)
            print("Finish energy statistics")

            statfile = os.path.join(self.hparams.etl_path, 'stats.json')
            print("Save statistics to {}".format(statfile))
            with open(statfile, 'w', encoding='utf-8') as f:
                stats = dict({
                    'mel': melspec_stats,
                    'pitch': pitch_stats,
                    'energy': energy_stats,
                })
                f.write(json.dumps(stats))
            print("Finish statistics computing!!!")

        return paths, stems

    def preprocess(self, path, stem, prep):
        wavfile = os.path.join(path, 'wav', f'{stem}.wav')
        wave, sampling_rate, bit_depth = load_wavefile(wavfile)

        if bit_depth != self.hparams.bit_depth:
            raise ValueError(f"{bit_depth} BD doesn't match target {self.hparams.bit_depth} BD for {wavfile} ")
        if sampling_rate != self.hparams.sampling_rate:
            if self.hparams.force_frame_rate:
                wave = librosa.resample(wave, orig_sr=sampling_rate, target_sr=self.hparams.sampling_rate)
                sampling_rate = self.hparams.sampling_rate
            else:
                raise  ValueError(f"{sampling_rate} SR doesn't match target {self.hparams.sampling_rate} SR for {wavfile} ")

        if prep.match_volume:
            wave = normalize_volume(wave, target_dBFS=-20)
        if prep.trim_silence: # don't enable, will destroy align
            wave, mask = trim_long_silences(wave,sampling_rate, bit_depth)


        melspec, magnitudes = self.stft.wave_to_melspec(wave)
        melspec = melspec.transpose((1,0)) # [T, Mel]

        # Durations
        if 'drns' in self.hparams.keywords: # W/O durations
            # Get duration via sampling points
            dur_path = os.path.join(path, 'wav', f'{stem}.txt')
            duration = get_duration_from_file(dur_path,self.hparams.hop_length)
            if sum(duration) != melspec.shape[0]:
                ValueError(f"Duration doesn't match melspec shape"
                           f"({sum(duration)} ,{melspec.shape[0]}) for {wavfile} ")
            duration = np.array(duration).astype(np.int64)

            os.makedirs(os.path.join(path, 'duration'), exist_ok=True)
            np.save(os.path.join(path, 'duration', f'{stem}.npy'), duration)

        # Pitch feature
        if 'f0s' in self.hparams.keywords: # W/O pitch feature
            ## Dio algorithm
            # pitch, times = pw.dio( # raw pitch extractor
            #     wave, self.hparams.sampling_rate,
            #     frame_period=self.hparams.hop_length / self.hparams.sampling_rate * 1000
            # )
            # pitch = pw.stonemask( # pitch refinement
            #     wave, pitch, times, self.hparams.sampling_rate
            # )
            ## Harvest algorithm
            # pitch, times = pw.harvest(# raw pitch extractor
            #     wave, self.hparams.sampling_rate,
            #     frame_period=self.hparams.hop_length / self.hparams.sampling_rate * 1000
            # )
            ## Yin algorithm
            pitch, voiced_mask, p_voiced = librosa.pyin(
                wave,fmin=self.hparams.f0_min, fmax=self.hparams.f0_max,
                sr=self.hparams.sampling_rate, frame_length=self.hparams.filter_length,
                win_length=self.hparams.filter_length // 2, hop_length=self.hparams.hop_length, fill_na=0
            )
            times = librosa.times_like(pitch)

            _ = interpolate(pitch) # interpolate pitch

            os.makedirs(os.path.join(path, 'pitch'), exist_ok=True)
            np.save(os.path.join(path, 'pitch', f'{stem}.npy'), pitch)

        # Energy feature
        if 'nrgs' in self.hparams.keywords: # W/O energy feature
            energy = np.linalg.norm(magnitudes, ord=2, axis=0)
            _ = interpolate(energy)  # interpolate energy

            os.makedirs(os.path.join(path, 'energy'), exist_ok=True)
            np.save(os.path.join(path, 'energy', f'{stem}.npy'), energy)

        # Save Mel-spectrogram
        os.makedirs(os.path.join(path, 'mel'), exist_ok=True)
        np.save(os.path.join(path, 'mel', f'{stem}.npy'), melspec)

