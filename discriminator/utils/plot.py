import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import  colorsys
import random
random.seed(1234)
import numpy as np
from umap.umap_ import umap
from skimage.io import imsave
from skimage import img_as_ubyte


def save_to_png(file_name, array):
    """Save the given numpy array as a PNG file."""
    # from skimage._shared._warnings import expected_warnings
    # with expected_warnings(['precision']):
    imsave(file_name, img_as_ubyte(array))


def save_figure(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data



def plot_alignment(alignment, title=None, max_len=None, save_path=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        np.rot90(alignment),
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)

    plt.title(title)

    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure(fig)
    plt.close()

    if save_path is not None:
        save_to_png(save_path, data)

    return fig, data


def plot_spectrogram(spectrogram, title=None, max_len=None, target=None, save_path=None):
    if max_len is not None:
        spectrogram = spectrogram[:max_len, :]


    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)

    im = ax1.imshow(np.rot90(spectrogram), aspect='auto', origin='lower', interpolation='none')
    ax1.set_title('Predicted Spectrogram')

    # Set common labels
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)
    fig.colorbar(mappable=im, shrink=0.65, ax=ax1)
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Channels")

    #target spectrogram subplot
    if target is not None:
        if max_len is not None:
            target = target[:max_len, :]

        ax2 = fig.add_subplot(212)

        im = ax2.imshow(np.rot90(target), aspect='auto', origin='lower', interpolation='none')
        ax2.set_title('Target Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, ax=ax2)
        ax2.set_xlabel("Frames")
        ax2.set_ylabel("Channels")

    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure(fig)
    plt.close()

    if save_path is not None:
        save_to_png(save_path, data)

    return fig, data

def get_hls_colors(count):
    hls_colors = []
    init = 0.0
    step = 360.0 / count
    for i in range(count):
        h = init + i * step
        l = 50.0 + random.random() * 10.0
        s = 90.0 + random.random() * 10.0
        hls_colors.append([h / 360.0, l / 100.0, s / 100.0])
    random.shuffle(hls_colors)
    return hls_colors

def get_rgb_colors(count):
    rgb_colors = []
    if count <1: return rgb_colors
    hls_colors = get_hls_colors(count)
    for hls in hls_colors:
        r, g, b = colorsys.hls_to_rgb(*hls)
        rgb_colors.append([int(r * 255), int(g * 255), int(b * 255)])
    random.shuffle(rgb_colors)
    return rgb_colors

# colormap=np.random.randint(0，256,size=(25,3)).astype(npfloat)/255
colormap=np.array(get_rgb_colors(10)).astype(np.float) /255

def plot_proiection(embeddings, utterances_per_voice, max_voices=10, save_path=None):
    embeddings = embeddings[:max_voices * utterances_per_voice]

    n_voices = len(embeddings) // utterances_per_voice
    ground_truth = np.repeat(np.arange(n_voices),utterances_per_voice)
    colors = [colormap[i] for i in ground_truth]


    reducer =umap.UMAP()
    projected =reducer.fit_transform(embeddings)

    fig, ax =plt.subplots(figsize=(6,4))
    ax.scatter(projected[:, 0], projected[:1], c=colors,marker='o', s=20,
                label='projections')
    # hide x&y axises
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure(fig)
    plt.close()

    if save_path is not None:
        save_to_png(save_path, data)

    return fig, data

if __name__ == "__main__":
    import numpy as np

    spec = np.random.randn(128, 80)
    spec = plot_spectrogram(spec, target=spec, save_path='spec.png')

    attn = np.random.randn(60, 120)
    attn = np.exp(attn) / np.exp(attn).sum(axis=1, keepdims=True)
    attn = plot_alignment(attn,save_path='attn.png')