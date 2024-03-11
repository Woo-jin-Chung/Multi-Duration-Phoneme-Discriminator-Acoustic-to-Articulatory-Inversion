import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import pyworld as pw
import scipy
import scipy.signal
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
#from acoustic_feature import feature_extract
import pdb
# import LPanalysis as LP
# import torchcrepe
import torch.nn.functional as F
import torchaudio
import json
import re
import logging
import augment
import scipy.signal as ss
import soundfile as sf


def get_dataset_filelist(spk):
    json_dir = '/path/to/json/'
    train_json = os.path.join(json_dir, f'train_{spk}.json')
    test_json = os.path.join(json_dir,  f'test_{spk}.json')
    with open(train_json, 'r') as f:
        train = json.load(f)
    with open(test_json, 'r') as f:
        test = json.load(f)

    return train, test


class Audioset:
    def __init__(self, files=None, length=None, stride=None, hop_size=None,
                pad=True, with_path=True, sample_rate=None,
                channels=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.hop_size = hop_size
        self.ema_hop = 160
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = int(self.length)
                out, sr = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)                
                filename = file.split('/')[-1].split('.')[0]
                ema_base = '/path/to/all/your/ema/data/'
                ema_file = os.path.join(ema_base, filename.split('_')[0], 'ema', filename + '.npy')
                ema = np.load(ema_file)
                ema = torch.FloatTensor(ema[int(offset/self.ema_hop):int(offset/self.ema_hop) + int(num_frames/self.ema_hop)]).permute(1,0)[:-1,:]
            else:
                out, sr = torchaudio.load(str(file), frame_offset=0, num_frames=-1)
                filename = file.split('/')[-1].split('.')[0]
                ema_base = '/path/to/all/your/ema/data/'
                ema_file = os.path.join(ema_base, filename.split('_')[0], 'ema', filename + '.npy')
                ema = np.load(ema_file)
                ema = torch.FloatTensor(np.load(ema_file))

                
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]

            if sr != target_sr:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_sr}, but got {sr}")
            if out.shape[0] != target_channels:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_channels}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
                ema = F.pad(ema, (0, int(num_frames/self.ema_hop) - ema.shape[-1]))

            if self.with_path: 
                return out, ema, file
            else:
                return out, ema


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, traindata, n_fft, num_mels,
                hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                device=None, fmax_loss=None, train=True, length=int(16000*(0.9+0.5)),#4.5*16000,
                stride=int(0.5*16000), pad=True):
        self.traindata = traindata
        random.seed(1234)
        if shuffle:
            random.shuffle(self.traindata)
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.cached_clean_wav = None
        self.n_cache_reuse = n_cache_reuse
        self.device = device
        self.train = train
        self.ema_hop = 160
        self.num_examples = []
        if split == True:
            self.length = length
        else:
            self.length = None
        self.stride = stride or length
        self.pad = pad

        if train:
            self.rir_list = pd.read_csv('/home/woojinchung/codefile/Naver2022/integrated/mit_rir_tr.csv')
        else:
            self.rir_list = pd.read_csv('/home/woojinchung/codefile/Naver2022/integrated/mit_rir_tt.csv')


        clean = self.traindata
        kw = {'length': self.length, 'stride': stride, 'pad': pad, 'sample_rate': sampling_rate, 'with_path': True, 'hop_size': hop_size}
        self.clean_set = Audioset(clean, **kw)

        ### shift augmentation
        augments = []
        shift = 8000
        self.shift = shift
        shift_same = True
        
        augments.append(augment.Shift(shift, shift_same))
        self.augment = torch.nn.Sequential(*augments)

    def __getitem__(self, index):        
        cleanaudio = self.clean_set[index][0]
        ema = self.clean_set[index][1]
        filename = self.clean_set[index][2].split('/')[-1].split('.')[0]

        if self.train:
            clean = cleanaudio
            # Shift augmentation
            clean = clean.unsqueeze(0)
            sources = torch.stack([clean, clean])
            sources, e_offsets = self.augment(sources)
            clean, clean = sources
            clean = clean.squeeze(0)
            cleanaudio = clean

            e_length = ema.size(-1) - int(self.shift / self.ema_hop)
            e_offsets = e_offsets.expand(1,-1)
            ema = ema[:,e_offsets:e_offsets+e_length]
            

        cleanaudio = torch.FloatTensor(cleanaudio)

        return (cleanaudio.squeeze(0), 
                ema.squeeze(0), index, filename)


    def __len__(self):
        return len(self.clean_set)
