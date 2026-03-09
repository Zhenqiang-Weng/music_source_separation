## 可参考  https://github.com/keonlee9420/PortaSpeech/blob/main/preprocessor/preprocessor.py
import numpy as np
from scipy.stats import  betabinom
import os
from tqdm import tqdm

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)

mel = np.load('mel.npy')
duration = np.load('dur.npy')
attn_prior = beta_binomial_prior_distribution(mel.shape[1],len(duration),1)


