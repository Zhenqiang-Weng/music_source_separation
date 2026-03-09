"""
Dataset class for speech dataset.
"""

import os
import json
import numpy as np

import torch
import torch.utils.data import Dataset

from helpers.processor import TextProcessor, Specprocessor
from utils.functional import interpolate, aggregate_by_duration
from utils.transform import Transform, ToTensor, Pad, Normalize, Quantize, Clamp


class SpeechDataset(Dataset):

    KEYS = ['texts', 'brks', 'lens', 'specs', 'drns', 'f0s', 'nrgs', 'spks', 'stls', 'attn', 'word']

    def __init__(self,filelist, hparams):
        self.hparams = hparams
        self.etl_path = hparams.etl_path
        self.keys = hparams.keywords if len(hparams.keywords) != 0 else self.KEYS
        self.hparams.keywords = self.keys

        self._text_processor = None
        self._spec_processor = None

        self.prepare(os.path.join(self.etl_path, filelist))

    def prepare(self, listfile):
        if not os.path.isfile(listfile):
            raise ValueError("File {} not exists".format(listfile))


        paths, texts, _, speakers, styles = self.load_metadata(listfile)

        ## Spectrograms
        if 'specs' in self.keys:
            if self._spec_processor is None:
                self._spec_processor = Specprocessor(self.hparams)
            paths, stems = self._spec_processor(paths)
        else:
            paths, stems = [ None ] * len(paths), paths

        ## Texts
        if 'texts' in self.keys:
            if self._text_processor is None:
                self._text_processor = TextProcessor(self.hparams)
            texts, breaks, lengths = self._text_processor(texts)
        else:
            texts = breaks = lengths = [ None ] * len(texts)


        ## Voices
        if 'spks' in self.keys:
            speakers = [[int(speaker)] for speaker in speakers]
        else:
            speakers = [ None ] * len(speakers)

        if 'stls' in self.keys:
            styles = [[int(style)] for style in styles]
        else:
            styles = [ None ] * len(styles)


        filter = lambda it: [x for x in it if False not in x]
        self.metadata = filter(zip(texts, breaks, lengths, paths, stems, speakers, styles))

        self.update(self.metadata[:])

    def update(self, metadata):
        self.texts, self.breaks, self.lengths, \
            self.paths, self.stems, self.speakers, self.styles = zip(*metadata)


    def slice(self, start, end):
        self.update(self.metadata[start:end])

    def subset(self, indices):
        self.update(self.metadata[indices])

    def load_metadata(self, filepath, split='|'):
        with open(filepath, 'r' ,encoding='utf-8') as fp:
            metadata = [line.strip().split(split) for line in fp]
        return zip(*metadata)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = dict()
        if 'paths' in self.keys:
            data['paths'] = self.paths[index]
        if 'stems' in self.keys:
            data['stems'] = self.stems[index]
        if 'texts' in self.keys:
            data['texts'] = np.array(self.texts[index])
        if 'brks' in self.keys:
            data['brks'] = np.array(self.breaks[index])
        if 'lens' in self.keys:
            data['lens'] = np.array(self.lengths[index])
        if 'specs' in self.keys:
            data['specs'] = np.load(os.path.join(self.paths[index], 'mel', f'{self.stems[index]}.npy'))
        if 'slens' in self.keys:
            data['slens'] = np.array([data['specs'].shape[0]])
        if 'drns' in self.keys:
            data['drns'] = np.load(os.path.join(self.paths[index], 'duration', f'{self.stems[index]}.npy'))
        if 'f0s' in self.keys:
            data['f0s'] = np.load(os.path.join(self.paths[index], 'pitch', f'{self.stems[index]}.npy'))
            data['f0s'] = interpolate(data['f0s'])
            # if self.hparams.feature_level == 'phoneme':
            #     data['f0s'] = aggregate_by_duration(data['f0s'], data['drns']) # phone-level average
        if 'nrgs' in self.keys:
            data['nrgs'] = np.load(os.path.join(self.paths[index], 'energy', f'{self.stems[index]}.npy'))
            data['nrgs'] = interpolate(data['nrgs'])
            # if self.hparams.feature_level == 'phoneme':
            #     data['nrgs'] = aggregate_by_duration(data['nrgs'], data['drns']) # phone-level average
        if 'attn' in self.keys:
            data['attn'] = np.load(os.path.join(self.paths[index], 'attn_prior', f'{self.stems[index]}.npy'))

        if 'word' in self.keys:
            data['word_boundaries'] = np.load(os.path.join(self.paths[index], 'phones_per_word', f'{self.stems[index]}.npy'))
            data['text_w_lens'] = np.array([data['word_boundaries'].shape[0]])

        if 'spks' in self.keys:
            data['spks'] = np.array(self.speakers[index])
        if 'stls' in self.keys:
            data['stls'] = np.array(self.styles[index])
        return data


class SpeechCollate():
    """
    Zero-pads model inputs and targets
    """
    def __init__(self, hparams, device= None):
        transforms = dict()
        transforms['none'] = [ ] # none operations
        transforms['int'] = [
            ToTensor(dtype=torch.long, device=device), Pad(0)
        ]
        transforms['float'] = [
            ToTensor(dtype=torch.float, device=device), Pad(0)
        ]
        smin, smax, smean, sstd = self.get_statistics(
            os.path.join(hparams.etl_path, hparams.statistic_file), key='mel'
        )
        transforms['specs'] = [
            ToTensor(dtype=torch.float, device=device),
            Normalize(mode=hparams.normalization, min=smin, max=smax,mean=smean,std=sstd),Pad(0)
        ]
        transforms['f0s'] = [
            ToTensor(dtype=torch.float, device=device),
            # MIDI notes: pitch = 12 * log2(f0 / 440) + 69
            Quantize(mode='log2', k=12.0, a=0.0,b=-36.3763), Clamp(floor=35, ceil=88), Pad(0)
        ]
        transforms['specs'] = [
            ToTensor(dtype=torch.float, device=device),
            Quantize(mode='log2', k=1.0, a=0.0,b=0.0), Clamp(floor=-10, ceil=10), Pad(0)
        ]

        self.pipes = {
            key: Transform(transforms[key]) for key in transforms.keys()
        }

    def get_statistics(self, statfile, key='mel'):
        with open(statfile, 'r', encoding='utf-8') as fp:
            statistics = json.load(fp)
        return statistics[key]

    def __call__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        :param batch: [texts,specs]
        :return:
        """
        keys = batch[0].keys()
        collated_batch = dict()
        prepared_batch = dict()

        for data in batch:
            for key in keys:
                tmp = collated_batch.get(key, [])
                collated_batch[key] = tmp + [data[key]]

        for key, data in collated_batch.items():
            if key in ['specs', 'f0s', 'nrgs']:
                prepared_batch[key] = self.pipes[key](data).squeeze(-1)
            elif key in ['texts', 'brks', 'lens', 'slens', 'drns', 'word_boundaries']:
                prepared_batch[key] = self.pipes['int'](data).squeeze(-1)
            elif key in ['spks', 'stls', 'text_w_lens']:
                prepared_batch[key] = self.pipes['int'](data).squeeze(-1)
            elif key in ['attn']:
                prepared_batch[key] = self.pipes['float'](data).squeeze(-1)
            else:
                prepared_batch[key] = self.pipes['none'](data)

        return prepared_batch