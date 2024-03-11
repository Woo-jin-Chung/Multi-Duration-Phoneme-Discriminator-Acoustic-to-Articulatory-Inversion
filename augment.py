# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import torch
from torch import nn


class Shift(nn.Module):
    """Shift."""

    def __init__(self, shift=8192, same=False):
        """__init__.
        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        #pdb.set_trace()
        sources, batch, channels, length = wav.shape
        # 2, B, 1, T  = [2,B,1,T]
        length = length - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                e_offsets = torch.randint(int(self.shift/160),[1 if self.same else sources, batch, 1, 1], device=wav.device)
                offsets = e_offsets*160
                offsets = offsets.expand(sources, -1, channels, -1)
                indexes = torch.arange(length, device=wav.device)
                # print(indexes, offsets)
                # print(indexes + offsets)
                wav = wav.gather(3, indexes + offsets)
        return wav, e_offsets[0][0][0]
