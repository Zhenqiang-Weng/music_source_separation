import torch
import os
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '..')
from utils.hparams import HParams
from data.dataset import SpeechDataset,SpeechCollate

def test_dataloader():
    hparams = HParams(os.path.join('..', 'config', 'default.yaml'))

    path_hparams =hparams.path
    data_hparams =hparams.data
    data_hparams.update(path_hparams)
    text_hparams, audio_hparams = hparams.text, hparams.audio
    data_hparams.update(text_hparams)
    data_hparams.update(audio_hparams)
    print(data_hparams)

    dataset = SpeechDataset(data_hparams)
    print(len(dataset))

    data_hparams.symbols = dataset._text_processor.symbols
    print(data_hparams)

    for data in dataset:
        print(data['texts'], data['lens'])
        break
    # device must be cpu before dataloader
    collate_fn = SpeechCollate(data_hparams, device=torch.device('cpu'))
    data_hparams.batch_size = 2
    dataloader = DataLoader(dataset, num_workers=1, shuffle=True,
                            batch_size=data_hparams.batchsize, pin_memory=False,
                            drop_last=False, collate_fn=collate_fn)
    print(len(dataloader))

    for batch in dataloader:
        print(batch['texts'],batch['lens'])
        break

if __name__=="__main__":
    test_dataloader()