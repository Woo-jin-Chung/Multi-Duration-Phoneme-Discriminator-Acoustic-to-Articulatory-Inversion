from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from model import Conformer, compute_mask_indices


class DPD(torch.nn.Module):
    def __init__(self, duration, n_head=1, conf_layer=12, use_spectral_norm=False):
        super(DPD, self).__init__()
        self.duration = duration
        self.ema_channel = 18
        self.encoder_num_heads=n_head
        self.encoder_dim = self.duration * self.encoder_num_heads
        self.encoder_layers = conf_layer
        self.channel_split_token = nn.Parameter(
            torch.FloatTensor(self.encoder_dim).uniform_()
        )
        self.channel_end_token = nn.Parameter(
            torch.FloatTensor(self.encoder_dim).uniform_()
        )
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.encoder_dim).uniform_()
        )
        
        self.mask_length = 5
        self.mask_prob = 0.15
        self.mask_selection = 'static'
        self.mask_other = 0
        self.no_mask_overlap = False
        self.mask_min_space = 1
        
        self.channel_split_indices = [3,6,9,12,15]
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.conformer_blocks = nn.ModuleList([Conformer(self.encoder_dim, self.encoder_dim, self.encoder_num_heads, num_buckets=180, has_relative_attention_bias=True if i==0 else False)
                                                for i in range(self.encoder_layers)
                                                ])
        self.conv_post = norm_f(Conv1d(self.encoder_dim, 1, 3, 1))
    
    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        return x, mask_indices
    
    def forward(self, x, mask=False):
        b, c, t = x.shape
        assert (c == self.ema_channel), "check the tensor size"
        
        channel_split_token = self.channel_split_token.unsqueeze(0).unsqueeze(0).expand(b,1,-1)
        channel_split_token = channel_split_token.repeat(1,1,90//self.encoder_dim)
        channel_end_token = self.channel_end_token.unsqueeze(0).unsqueeze(0).expand(b,1,-1)
        channel_end_token = channel_end_token.repeat(1,1,90//self.encoder_dim)
        for idx in sorted(self.channel_split_indices, reverse=True):
            x = torch.cat((x[:,:idx], channel_split_token, x[:,idx:]), dim=1)
        
        x = torch.cat((x, channel_end_token), dim=1)
        
        # EMA reshaping
        x = x.view(b, c + len(self.channel_split_indices) + 1, -1, self.duration)
        x = x.permute(0,2,1,3)
        x = x.contiguous().view(b, -1, self.duration)

        if self.encoder_num_heads != 1:
            x = x.repeat(1,1,self.encoder_num_heads)

        fmap=[]
        if mask:
            x, mask_indices = self.apply_mask(
                x, padding_mask=None)
        
        pos_bias = None
        for ii, conf_block in enumerate(self.conformer_blocks):
            x, pos_bias = conf_block(x, pos_bias=pos_bias)
            fmap.append(x)
        x = self.conv_post(x.permute(0,2,1))
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    
class MDPD(torch.nn.Module):
    def __init__(self, n_head=1, conf_layer=6):
        super(MDPD, self).__init__()
        self.discriminators = nn.ModuleList([
            DPD(6, n_head=n_head, conf_layer=conf_layer),
            DPD(9, n_head=n_head, conf_layer=conf_layer),
            DPD(10, n_head=n_head, conf_layer=conf_layer),
            DPD(15, n_head=n_head, conf_layer=conf_layer),
            DPD(18, n_head=n_head, conf_layer=conf_layer),
        ])

    def forward(self, y, y_hat, mask=False):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, mask=mask)
            y_d_g, fmap_g = d(y_hat, mask=mask)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    
