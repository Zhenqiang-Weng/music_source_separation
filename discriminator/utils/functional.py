import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d


def mask(shape, lengths, dim=-1):

    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == shape[0], 'Lengths must contain as many elements as there are items in the batch'

    lengths = torch.as_tensor(lengths)

    to_expand = [1] * (len(shape)-1)+[-1]
    mask = torch.arange(shape[dim]).expand(to_expand).transpose(dim, -1).expand(shape).to(lengths.device)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask


def positional_encoding(channels, length=2048, w=1):
    """The positional encoding from `Attention is all you need` paper

    :param channels: How many channels to use
    :param length:
    :param w: Scaling factor
    :return:
    """
    pe = torch.FloatTensor(length, channels)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.agrange(
        0, channels, 2).float() * (-math.log(10000.0) / channels
    ))
    pe[:, 0::2] = torch.sin(w * position * div_term)
    pe[:, 1::2] = torch.cos(w * position * div_term)
    return pe


def create_positions(durations, mode='duration'):
    B, N = durations.shape
    T = torch.max(torch.sum(durations, dim=-1)).item()

    positions = torch.zeros(B, T)
    for i in range(B):
        count = 0
        for j in range(N):
            for k in range(durations[i][j]):
                if mode == 'duration':
                    positions[i][count+k] = k+1
                elif mode == 'standard':
                    positions[i][count+k] = count + k + 1
                else:
                    assert False, "Invalid mode"
            count += durations[i][j]
    return positions

def scaled_dot_attention(q, k, v, mask=None, noise=0, dropout=lambda x: x):
    """
    :param q: queries, (batch, time1, channels1)
    :param k: keys, (batch, time2, channels1)
    :param v: values, (batch, time2, channels2)
    :param mask: boolean mask, (batch, time1, time2)
    :param dropout: a dropout function - this allows keeping dropout as a module -> better control when training/eval
    :return: (batch, time1, channels2), (batch, time1, time2)
    """

    # (batch, time1, time2)
    weights = torch.matmul(q, k.transpose(2, 1))
    if mask is not None:
        weights = weights.masked_fill(~mask, float('-inf'))

    if noise:
        weights += noise * torch.randn(weights.shape).to(weights.device)

    weights = torch.softmax(weights, dim=-1)
    weights = dropout(weights)

    result = torch.matmul(weights, v)  # (batch, time1, channels2)
    return result, weights


def get_durations_from_alignment(alignment):
    """A list of alignment
    :param alignment: list of tensors, shape (batch, slens, tlens)
    :return durations: list of tensors, shape (batch, tlens)
    """

    durations=list()
    for align in alignment:
        duran = torch.zeros(align.shape).to(align.device)
        t = torch.arange(align.shape[0]).to(align.device)
        maxa = torch.max(align,dim=1)[1]
        duran[t, maxa] = 1.0
        duran = torch.sum(duran, dim=0).long()
        durations.append(duran)

    return durations

def get_alignment_from_durations(durations):
    """Map list of durations to alignment matrix Allows backwards mapping for s nity check.
    :param durations: list of tensors, shape (batch, tlens)
    :return alignment: list of tensors, shape (batch, slens, tlens)
    """

    alignment = list()
    for duran in durations:
        frames = duran.long().sum()
        align = torch.zeros((frames, duran.shape[0])).to(duran.device)
        x = torch.arange(frames).to(duran.device)
        # repeat each symbols index according to durations
        y = torch.repeat_interleave(torch.arange(duran.shape[0]).to(duran.device), duran.long())
        align[x, y] =1.0
        alignment.append(align)

    return alignment

def pad_batch(items,pad_value=0):
    """Pad tensors in list to equal length
    :param items:
    :param pad_value:
    :return: padded_items, lengths
    """
    max_lens = [max([item.shape[i] for item in items]) for i in range(items[0].ndim)]

    padded_items = list()
    for item in items:
        pad_size = [t - c for t,c in zip(max_lens, list(item.shape))]
        pad_size = len(pad_size) * [0] + pad_size[::-1]
        pad_size = pad_size[0::2] + pad_size[1::2]
        padded_item = F.pad(torch.as_tensor(item), pad=pad_size, value=pad_value)
        padded_items.append(padded_item)

    padded_items = torch.stack(padded_items)

    origin_lengths = [torch.tensor([item.shape[i] for item in items]) for i in range(items[0].ndim)]

    return padded_items, origin_lengths

def unpad_batch(items, lengths):
    """Unpad tensors (batch) to tensors in list with given length
    param items::param lengths:
    :return: unpadded_items
    """

    max_lens = list(items[0].shape)

    unpadded_items=list()
    for idim, item in enumerate(items):
        origin_lens = [ll[idim] for ll in lengths]
        unpad_size = [t-c for t,c in zip(origin_lens, max_lens)]
        unpad_size = len(unpad_size) * [0] + unpad_size[::-1]
        unpad_size = unpad_size[0::2] + unpad_size[1::2]
        unpadded_item = F.pad(torch.as_tensor(item), pad=unpad_size, value=0)
        unpadded_items.append(unpadded_item)


    return unpadded_items

def warp_duration(durn_f, durn_i):
    total_diff = sum(durn_f) - sum(durn_i)

    drop_diffs = np.array(durn_f) - np.array(durn_i)
    drop_order = np.argsort(-drop_diffs)
    for i in range(math.floor(total_diff) + 1):
        index = drop_order[i]
        durn_i[index] += 1
    return durn_i

def get_duration_from_file(filename, hop_length):
    with open(filename, "r") as f:
        lines = f.readlines()
    durn_f, durn_i = list(), list()
    durn_sum = 0.0
    for line in lines:
        if line.strip() == '': continue
        phone, durn = line.strip().split("|")
        durn_sum += float(durn)
        durn = float(durn) / hop_length
        durn_f.append(durn)
        durn_i.append(int(durn))
    # duration correction
    durn_sum /= hop_length
    durn_f.pop()
    durn_f.append(durn_sum - sum(durn_f))
    # duration warping
    durn_i = warp_duration(durn_f, durn_i)

    return durn_i

def interpolate(feature):
    valids = np.where(feature != 0)[0]
    assert len(valids) >= 2, "Non-zero sample points less than 2"
    interp_fn = interp1d(
        valids,
        feature[valids],
        fill_value = (feature[valids[0]], feature[valids[-1]]),
        bounds_error = False,)
    feature = interp_fn(np.arange(0, len(feature)))

    return feature

def aggregate_by_duration(feature, duration, interpo=False):
    """aggregate feature according to given duration. Args:
    feature (np.ndarray): Feature contour extracted from pyworld. duration (List[int]): List of durations. Returns:
    np.ndarray: Preprocessed feature.
    """
    feature = feature[:sum(duration)]
    if interpo:
        # interpolate zero-frames
        feature = interpolate(feature)

    # compute feature per phoneme
    feature_per_phoneme = list()
    start = 0
    for d in duration:
        v = np.mean(feature[start:start + d]) if d > 0 else 0
        feature_per_phoneme.append(v)
        start += d
    feature = np.array(feature_per_phoneme)
    return feature


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return ~mask

if	__name__ == "__main__":
    x = torch.randn(2, 4, 3)
    shape = x.size()
    lengths = torch.tensor([4,3])
    msk =mask(shape, lengths, dim=1)
    print(msk)

    pe =positional_encoding(2,64)
    print(pe[::8,:])
    durations =[torch.tensor([1,2,1]), torch.tensor([2,3])]
    alignment = get_alignment_from_durations(durations)
    alignment, origin_lens = pad_batch(alignment)
    print(alignment)

    alignment = unpad_batch(alignment,origin_lens)
    print(alignment)

    durations = get_durations_from_alignment(alignment)
    print(durations)

    durations, _ = pad_batch(durations)
    positions = create_positions(durations)
    print(positions)

    feature = np.random.randn(8)
    print(feature)
    duration = np.array([1, 3, 2, 2])
    feature = aggregate_by_duration(feature, duration)
    print(feature)