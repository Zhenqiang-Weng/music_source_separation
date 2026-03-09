import torch
import os
import sys
sys.path.insert(0, '..')
from utils.util import select_device, get_symbols
from utils.hparams import HParams
from helpers.trainer import CCTrainer

def create_hparams(hparams):
    # Data HParams
    data_hparams = hparams.data
    text_hparams, audio_hparams = hparams.text, hparams.audio
    symbols_list = os.path.join(data_hparams.etl_path, 'symbols.list')
    text_hparams.symbols, _ = get_symbols(symbols_list)
    text_hparams.symbols =\
        text_hparams.specials + text_hparams.symbols + text_hparams.punctuations
    data_hparams.update(text_hparams)
    data_hparams.update(audio_hparams)
    # Model HParams
    model_hparams =hparams.model
    model_hparams.n_symbols = len(text_hparams.symbols)
    # Train HParams
    train_hparams = hparams.train
    train_hparams.name = hparams.name
    return hparams
def test_trainer():
    hparams = HParams(os.path.join('..', 'config', 'default.yaml'))
    hparams = create_hparams(hparams)
    device = select_device(str(0))
    CCTrainer(hparams,device=device).fit()

if __name__ == "__main__":
    test_trainer()