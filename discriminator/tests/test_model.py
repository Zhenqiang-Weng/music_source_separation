import torch
import os
import sys
sys.path.insert(0, '..')
from utils.hparams import HParams
from utils.util import get_symbols
from models import CC

def test_CC():
    hparams = HParams(os.path.join('..', 'config', 'default.yaml'))

    path_hparams = hparams.path
    model_hparams = hparams.model
    model_hparams.update(path_hparams)
    symbols_list = os.path.join(model_hparams.out_path, 'symbols.list')
    symbols, _ = get_symbols(symbols_list)
    text_hparams = hparams.text
    text_hparams.symbols = symbols
    text_hparams.symbols =\
        text_hparams.specials + text_hparams.symbols + text_hparams.punctuations
    model_hparams.n_symbols = len(text_hparams.symbols)

    model = CC(model_hparams)
    # print(model)

    ilen = 25 # the min text length
    texts = torch.ones(2, ilen).long()
    durations = torch.randint(1, 3, (2, ilen)).long()
    lengths = durations.sum(dim=-1)
    alpha = None
    mels = torch.ones(2, lengths.max(), 80).float()

    inputs = (texts, lengths, durations, alpha)
    melspecs, prd_durans = model(inputs)
    print(melspecs.size())
    print(prd_durans.size())

if __name__ == "__main__":
    test_CC()
