import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import functools
import numpy as np
import librosa
import torch.nn as nn
import pdb

def plot_ema_compared(ema_hat, ema):
    ch, length = ema.shape
    fig, ax = plt.subplots(ch,1,figsize=(20,30))
    for i in range(ch):
        ax[i].plot(ema_hat[i], color='b', label='est')
        ax[i].plot(ema[i], color='r', label='gt')
        ax[i].legend()
    fig.canvas.draw()
    plt.close()
    return fig

def plot_spectrogram(spectrogram, frame_shift=60):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none', cmap = 'jet')#vmin = -80, vmax = 0)
    plt.colorbar(im, ax=ax)
    plt.xlim(0+frame_shift, 150+frame_shift)

    fig.canvas.draw()
    plt.close()

    return fig

def mel_for_plot(wav, sampling_rate):
    stft = librosa.stft(wav.cpu().numpy(), n_fft = 2048, win_length = 512, hop_length = 160, window = 'hamming')
    mag, _ = librosa.magphase(stft)
    mel_spec = librosa.feature.melspectrogram(S = mag, sr=sampling_rate, n_fft=2048, hop_length=160, n_mels=256, fmax=8000)
    dBstft = librosa.amplitude_to_db(mel_spec, ref = np.max)
    return torch.tensor(dBstft)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def teacher_ema(student_model: nn.Module, teacher_model: nn.Module, current_step: int, end_step: int):
    student_model.eval()
    teacher_model.eval()
    tau_start = 0.99
    tau_end = 0.99999
    
    if current_step >= end_step:
        tau = tau_end
    else:
        tau = (current_step / end_step) * (tau_end - tau_start) + tau_start
    
    for teacher_model, student_model in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_model.data = teacher_model.data * tau + student_model.data * (1. - tau)
    