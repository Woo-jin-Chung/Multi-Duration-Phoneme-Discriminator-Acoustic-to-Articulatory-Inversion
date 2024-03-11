import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from module import LayerNorm, MultiheadAttention, Conv1dStaticSamePadding,\
                    Swish, P_Conv, NP_Conv
import numpy as np


class ModelConfig:
    def __init__(self, cfg=None):
        # masking
        self.mask_length: int = 5     # mask length
        self.mask_prob: float = 0.15     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # relative position embedding
        self.relative_position_embedding: bool = True     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 800     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = True     # apply gated relative position embedding
        
        # mel conformer
        self.conformer_layers: int = 8
        self.conformer_attention_heads: int = 8
        self.conformer_embed_dim: int = 256
        self.ema_channel : int = 18
        self.wavlm_emb_dim : int = 1024

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class ArticulatoryInverter(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        p_factors = [5, 7, 11, 13]
        np_factor = 0.2
        
        self.cfg = cfg
        self.specaug = FbankAug()
        self.wavlm_emb_dim = cfg.wavlm_emb_dim
        self.mask_length = cfg.mask_length
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.conformer_embed_dim).uniform_()
        )
        self.post_feat_proj = nn.Linear(self.wavlm_emb_dim, cfg.conformer_embed_dim) if self.wavlm_emb_dim != cfg.conformer_embed_dim else None
        self.layer_norm = nn.LayerNorm(cfg.wavlm_emb_dim)
        
        self.pnp_conformer_blocks = nn.ModuleList([PNPConformer(cfg.conformer_embed_dim, cfg.conformer_embed_dim, cfg.conformer_attention_heads, p_factors[i], np_factor, has_relative_attention_bias=True if i==0 else False)
                                                for i in range(cfg.conformer_layers//2)
                                                ])
        self.conformer_blocks = nn.ModuleList([Conformer(cfg.conformer_embed_dim, cfg.conformer_embed_dim, cfg.conformer_attention_heads, has_relative_attention_bias=True if i==0 else False)
                                                for i in range(cfg.conformer_layers//2)
                                                ])
        self.ema_proj = nn.Linear(cfg.conformer_embed_dim, cfg.ema_channel)

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
    
    def forward(self, signal, mask=False, specaug=False):
        
        if specaug==True:
            with torch.no_grad():
                signal = self.specaug(signal)
        
        features = self.layer_norm(signal)

        if self.post_feat_proj is not None:
            features = self.post_feat_proj(features) 

        if mask:
            features, mask_indices = self.apply_mask(
                features, padding_mask=None)
            
        pos_bias=None
        for ii, pnp_conf_block in enumerate(self.pnp_conformer_blocks):
            features, pos_bias = pnp_conf_block(features, pos_bias)
        for ii, conf_block in enumerate(self.conformer_blocks):
            features, pos_bias = conf_block(features, pos_bias)
            # features = mel_hat
        ema_hat = self.ema_proj(features)
                    
        return ema_hat.permute(0,2,1)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x
    
class FeedFowardBlock(nn.Module):
    def __init__(self, in_channels):
        super(FeedFowardBlock, self).__init__()
        self.layer_norm = LayerNorm(in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels * 4)
        self.activation = Swish()
        self.linear2 = nn.Linear(in_channels * 4, in_channels)

    def forward(self, x):
        # x : [B, T, C]
        org = x
        x = self.layer_norm(x)
        x = self.activation(self.linear1(x))
        x = F.dropout(x, p=0.1)
        x = self.linear2(x)
        x = F.dropout(x, p=0.1) + org
        
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, has_relative_attention_bias=True, num_buckets=320, max_distance=800):
        super(MultiHeadAttentionBlock, self).__init__()
        self.layer_norm = LayerNorm(hidden_dim)
        self.self_attn_with_relpos = MultiheadAttention(
                                                    hidden_dim,
                                                    n_heads,
                                                    dropout=0.1,
                                                    self_attention=True,
                                                    has_relative_attention_bias=has_relative_attention_bias,
                                                    num_buckets=320, # number of buckets for relative position embedding
                                                    max_distance=800, # maximum distance for relative position embedding
                                                    gru_rel_pos=True, # apply gated relative position embedding as WavLM
                                                    rescale_init=False,
                                                    )
        
    def forward(self, x, pos_bias = None):
        org = x
        x = self.layer_norm(x)
        x, attn, pos_bias = self.self_attn_with_relpos(
                query=x,
                key=x,
                value=x,
                key_padding_mask=None,
                need_weights=False, # for optimized mha, if true returns the attn_weights instead of None
                attn_mask=None,
                position_bias=pos_bias
            )
        x = F.dropout(x, p=0.1) + org
        
        return x, pos_bias
    
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=5):
        super(ConvolutionBlock, self).__init__()
        
        self.resconv_bool = False
        if out_channels is None:
            out_channels = in_channels
        if out_channels != in_channels:
            self.resconv_bool = True
            self.residual_conv = Conv1dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.layer_norm = LayerNorm(in_channels)
        self.pointwise_conv1 = Conv1dStaticSamePadding(in_channels, out_channels*2, kernel_size=1, stride=1)
        self.glu_act = nn.GLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.depthwise_conv = Conv1dStaticSamePadding(out_channels, out_channels,
                                                        kernel_size=kernel_size, stride=1, groups=in_channels, bias=False)
        self.swish_act = Swish()
        self.pointwise_conv2 = Conv1dStaticSamePadding(out_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        # [BTC]
        org = x
        x = self.layer_norm(x).permute(0,2,1)
        # [BCT]
        x = self.glu_act(self.pointwise_conv1(x).permute(0,2,1)).permute(0,2,1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish_act(x)
        x = self.pointwise_conv2(x).permute(0,2,1)
        # [BTC]
        if self.resconv_bool:
            x = F.dropout(x, p=0.1) + self.residual_conv(org.permute(0,2,1)).permute(0,2,1)
        else:
            x = F.dropout(x, p=0.1) + org
        
        return x

class PNPConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, pfactor=5, npfactor=0.2):
        super(PNPConvolutionBlock, self).__init__()
        
        self.resconv_bool = False
        if out_channels is None:
            out_channels = in_channels
        if out_channels != in_channels:
            self.resconv_bool = True
            self.residual_conv = Conv1dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
            
        self.layer_norm = LayerNorm(in_channels)
        self.pointwise_conv1 = Conv1dStaticSamePadding(in_channels, out_channels*2, kernel_size=1, stride=1)
        self.glu_act = nn.GLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.depthwise_conv_p = P_Conv(out_channels, out_channels,
                                                        kernel_size=5, stride=1, snakefactor=pfactor, groups=in_channels, bias=False)
        self.depthwise_conv_np = NP_Conv(out_channels, out_channels,
                                                        kernel_size=5, stride=1, snakefactor=npfactor, groups=in_channels, bias=False)
        self.swish_act = Swish()
        self.pointwise_conv2 = Conv1dStaticSamePadding(out_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        # [B,T,C]
        org = x
        x = self.layer_norm(x).permute(0,2,1)
        # [BCT]
        x = self.glu_act(self.pointwise_conv1(x).permute(0,2,1)).permute(0,2,1)
        # PNP conv operation (instead of normal depth-wise)
        x_p = self.depthwise_conv_p(x)
        x_n = self.depthwise_conv_np(x)
        x = x_p + x_n
        x = self.batch_norm(x)
        x = self.swish_act(x)
        x = self.pointwise_conv2(x).permute(0,2,1)
        # [BTC]
        if self.resconv_bool:
            x = F.dropout(x, p=0.1) + self.residual_conv(org.permute(0,2,1)).permute(0,2,1)
        else:
            x = F.dropout(x, p=0.1) + org
        
        return x

class Conformer(nn.Module):
    def __init__(self, in_ch, out_ch, n_heads, kernel_size=5, has_relative_attention_bias=True, num_buckets=320, max_distance=800):
        super().__init__()
        
        self.resconv_bool = False
        if in_ch != out_ch:
            self.resconv_bool = True
            self.residual_conv = Conv1dStaticSamePadding(in_ch, out_ch, kernel_size=1, stride=1)
        
        # Feed foward module 1
        self.ff_module1 = FeedFowardBlock(in_ch)
        # MHA
        self.mha_module = MultiHeadAttentionBlock(in_ch, n_heads, has_relative_attention_bias, num_buckets, max_distance)
        # Convolution module
        self.conv_module = ConvolutionBlock(in_ch, out_ch, kernel_size)
        # Feed foward module 2
        self.ff_module2 = FeedFowardBlock(out_ch)
        
        self.layernorm = LayerNorm(out_ch)

    def forward(self, x, pos_bias=None):
        ff_out1 = self.ff_module1(x)
        x = x + ff_out1/2
        
        mha_out, pos_bias = self.mha_module(x, pos_bias=pos_bias)
        x = x + mha_out
        
        conv_out = self.conv_module(x)
        if self.resconv_bool:
            x = self.residual_conv(x.permute(0,2,1)).permute(0,2,1)
        x = x + conv_out

        ff_out2 = self.ff_module2(x)    
        x = x + ff_out2/2
        
        x = self.layernorm(x)
        
        return x, pos_bias
        
class PNPConformer(nn.Module):
    def __init__(self, in_ch, out_ch, n_heads=8, p_factor=5, np_factor=0.2, has_relative_attention_bias=True, num_buckets=320, max_distance=800):
        super().__init__()
        
        self.resconv_bool = False
        if in_ch != out_ch:
            self.resconv_bool = True
            self.residual_conv = Conv1dStaticSamePadding(in_ch, out_ch, kernel_size=1, stride=1)
        
        # Feed foward module 1
        self.ff_module1 = FeedFowardBlock(in_ch)
        # MHA
        self.mha_module = MultiHeadAttentionBlock(in_ch, n_heads, has_relative_attention_bias, num_buckets, max_distance)
        # Convolution module
        self.conv_module = PNPConvolutionBlock(in_ch, out_ch, p_factor, np_factor)
        # Feed foward module 2
        self.ff_module2 = FeedFowardBlock(out_ch)
        
        self.layernorm = LayerNorm(out_ch)

    def forward(self, x, pos_bias=None):

        ff_out1 = self.ff_module1(x)
        x = x + ff_out1/2
        
        mha_out, pos_bias = self.mha_module(x, pos_bias=pos_bias)
        x = x + mha_out
        
        conv_out = self.conv_module(x)
        if self.resconv_bool:
            x = self.residual_conv(x.permute(0,2,1)).permute(0,2,1)
        x = x + conv_out

        ff_out2 = self.ff_module2(x)    
        x = x + ff_out2/2
        
        x = self.layernorm(x)
        
        return x, pos_bias