"""Wrapper synthesizer class for synthesizing audio."""

import sys
import os
import time
import json
from tqdm import tqdm
import math
import numpy as np
import torch
from torch.utils.data import DataLoader




from utils.functional import mask
from utils.transform import Normalize
from utils.tuil import get_symbols

from data.dataset import SpeechDataset, SpeechCollate
from models import CC



class Synthesizer:
    def __init__(self, model=None, checkpoint=None, testset=None, normalize=None, device=None):
        # model
        self.model = model
        self.testset = testset
        self.normalize = normalize

        # device
        self.device = device
        self.model.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch, self.step = 0, 0
        if checkpoint is not None:
            try:
                self.load_checkpoint(checkpoint)
            except Exception as err:
                print(err)

    def to_device(self, device):
        print(f'Sending model to {device}')
        self.device = device
        self.model.to(device)
        return self

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model'])
        print("Loaded checkpoint epoch=%d step=%d" % (self.epoch, self.step))

        return self

    def testloader(self, dataset, batch_size, collate_fn, num_workers=8, **kwargs):
        return DataLoader(dataset,batch_size=batch_size, collate_fn=collate_fn,
                          num_workers=0 if sys.platform.startswith('win') else num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, **kwargs)

class CCSynthesizer(Synthesizer):
    def __init__(self, hparams, device=torch.device('cuda')):
        self._create_hparams(hparams)

        model = CC(self.model_hparams).eval()
        checkpoint = self.infer.hparams.chkpt_path

        testset = SpeechDataset(self.data_hparams.testing_files, self.data_hparams)

        filepath = os.path.join(self.data_hparams.etl_path, self.data_hparams.statistic_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            smin, smax, smean, sstd = json.load(f)['mel']

        normalize = Normalize(mode=self.data_hparams.normalization, min=smin, max=smax,mean=smean,std=sstd)

        super(CCSynthesizer, self).__init__(model=model,
                                            checkpoint=checkpoint,
                                            testset=testset,
                                            normalize=normalize,
                                            device=device)

    def _create_hparams(self, hparams):
        # Data Hparams
        self.data_hparams = hparams.data
        self.data_hparams.keywords = ['stems', 'texts', 'brks', 'lens', 'spks', 'stls']
        text_hparams, audio_hparams = hparams.text, hparams.audio
        symbols_list = os.path.join(hparams.data.etl_path, hparams.data.symbols_list)
        text_hparams.symbols, _ = get_symbols(symbols_list)
        text_hparams.symbols = text_hparams.specials + text_hparams.symbols + text_hparams.punctuations
        self.data_hparams.update(text_hparams)
        self.data_hparams.update(audio_hparams)
        # Model HParams
        self.model_hparams = hparams.model
        self.model_hparams.n_symbols = len(text_hparams.symbols)
        # Infer HParams
        self.infer_hparams = hparams.infer

    def synthesize(self, checkpoint=None, textfile=None, generating=True, speed_m=1, pitch_a=0.0, energy_a=0.0):
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        os.makedirs(self.infer_hparams.specs_path, exist_ok=True)

        if generating:
            import soundfile as sf
            from helpers.processor import MySTFT

            waves_path = self.infer_hparams.specs_path.replace('mel', 'wav')
            os.makedirs(waves_path, exist_ok=True)
            # stft
            stft = MySTFT(self.data_hparams)

        if textfile is not None:
            self.testset.prepare(textfile) # re-prepare testset
        collate_fn = SpeechCollate(self.data_hparams, device=torch.device('cpu'))
        test_loader = self.testloader(self.testset,\
                                      batch_size=self.infer_hparams.batch_size, collate_fn=collate_fn)

        for it, batch in enumerate(test_loader):
            print("Batch: {}".format(it))

            spec_outputs, output_lengths = self.inference(batch, controls=(speed_m,pitch_a,energy_a))

            for index, stem in enumerate(batch['stems']):
                spk, stl = batch['spks'][index], batch['stls'][index]
                length = output_lengths[index].item()
                spec_output = spec_outputs[index][:, :length].cpu().numpy()

                if self.model_hparams.multi_speakers:
                    stem = f'{stem}.{spk}'
                if self.model_hparams.multi_styles:
                    stem = f'{stem}.{stl}'
                np.save(os.path.join(self.infer_hparams.specs_path, f'{stem}.npy'), spec_output.T)

                if generating:
                    wave = stft.melspec_to_wave(spec_output, griffin_iters=60)
                    sf.write(os.path.join(waves_path, f'{stem}.wav'), wave ,self.data_hparams.sampling_rate)
    def inference(self, batch, controls):
        # 1 --> speed
        inputs = self.model.parse_infer_batch(batch,controls, self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        spec_outputs, output_lengths, duration, features, _ = outputs
        if self.normalize is not None:
            spec_outputs = self.normalize(spec_outputs, inverse=True)
        msk = mask(spec_outputs.shape, output_lengths, dim=1).to(self.device)
        spec_outputs = spec_outputs.masked_fill(~msk, -11.5129).permute(0, 2, 1)

        return spec_outputs, output_lengths