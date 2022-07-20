from collections import OrderedDict
import re
from typing import Callable, Tuple, Union
import logging
from matplotlib.pyplot import get
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
import os
from pathlib import Path

# >>>>>>> Added for gumbel softmax <<<<<<<<<
from torch.autograd import Variable
# >>>>>>> Added for new attention module <<<<<<<<<
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from transformers import AutoModel

from timm.models.layers import DropPath, trunc_normal_

from utils.comm import comm
from utils.comm import gather_tensors

# >>>>>>> Added for conv reshape <<<<<<<<<
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, LayerNorm)):
            if comm.is_main_process():
                logging.info('=> init {} gamma to 1'.format(m))
                logging.info('=> init {} beta to 0'.format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from tranc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# For spottune before 0703
def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


# For spottune before 0703
def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


# For spottune before 0703
# def gumbel_softmax(logits, temperature = 5):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature)
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     return (y_hard - y).detach() + y

class Attention_CUST(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, config_additional=None,
                 modality=None, convit_layer_flag=False, cvt_flag=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Attention_CUST, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self._reset_parameters()

        self.modality = modality
        assert modality == 'visual' or modality == 'text'

        # >>>>>>>>>>> For output attention weight <<<<<<<<<<
        self.output_raw_attention_weight = getattr(config_additional, 'OUTPUT_ATTN_RAW', False)

        # >>>>>>>>>>> For CVT <<<<<<<<<<<<<
        self.cvt_flag = cvt_flag
        self.cvt_v_kernel = getattr(config_additional, 'CVT_V_KERNEL', False)
        self.cvt_v_stride = getattr(config_additional, 'CVT_V_STRIDE', False)
        self.cvt_v_pad = getattr(config_additional, 'CVT_V_PAD', False)
        if self.cvt_flag:
            self.cvt_q = None
            self.cvt_k = None
            self.cvt_v = None
            if getattr(config_additional, 'CVT_INSIDE_Q', False):
                self.cvt_q = self.build_cvt_dw(embed_dim)
            if getattr(config_additional, 'CVT_INSIDE_K', False):
                self.cvt_k = self.build_cvt_dw(embed_dim)
            if getattr(config_additional, 'CVT_INSIDE_V', False):
                self.cvt_v = self.build_cvt_dw(embed_dim)

        # >>>>>>>>>>> For CONVIT <<<<<<<<<<<<<
        self.convit_v_flag = getattr(config_additional, 'CONVIT_IN_V', False) and (
                    self.modality == 'visual') and convit_layer_flag
        self.convit_local_strength = getattr(config_additional, 'CONVIT_LOCAL_STRENGTH', 1)
        if self.convit_v_flag:
            self.convit_pos_proj = nn.Linear(3, self.num_heads)
            self.convit_gating_param = nn.Parameter(torch.ones(self.num_heads))

        # >>>>>>>>>>> For CONTAINER <<<<<<<<<<<<<
        self.container_v_flag = getattr(config_additional, 'CONTAINER_IN_V', False) and (self.modality == 'visual')
        self.container_v_kernel = getattr(config_additional, 'CONTAINER_V_KERNEL', 3)
        self.container_v_stride = getattr(config_additional, 'CONTAINER_V_STRIDE', 1)
        self.container_v_pad = getattr(config_additional, 'CONTAINER_V_PAD', 1)
        if self.container_v_flag:
            self.container_conv = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=self.container_v_kernel,
                    padding=self.container_v_pad,
                    stride=self.container_v_stride,
                    bias=False,
                    groups=embed_dim
                )),
                ('bn', nn.BatchNorm2d(embed_dim)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
            self.container_gating = nn.Parameter(torch.Tensor([0.0]))

        # >>>>>>>>>>> For LORA <<<<<<<<<<<<<
        self.lora_attn_dim = getattr(config_additional, 'LORA_ATTN_DIM', 0)
        self.lora_add_where = getattr(config_additional, 'LORA_WHERE_ADD', 'v0')

        if modality == 'visual' and getattr(config_additional, 'VISUAL_LORA_LOCAL', False):
            self.lora_local_visual = True
            self.kernel_size = getattr(config_additional, 'VISUAL_LORA_LOCAL_KERNEL', 3)
            self.padding_q = getattr(config_additional, 'VISUAL_LORA_LOCAL_PAD_Q', 1)
            self.padding_kv = getattr(config_additional, 'VISUAL_LORA_LOCAL_PAD_KV', 1)
            self.stride_q = getattr(config_additional, 'VISUAL_LORA_LOCAL_STRIDE_Q', 1)
            self.stride_kv = getattr(config_additional, 'VISUAL_LORA_LOCAL_STRIDE_KV', 1)
        else:
            self.lora_local_visual = False

        if self.lora_attn_dim > 0:
            self.lora_attn_alpha = getattr(config_additional, 'LORA_ATTN_ALPHA', 0)
            self.lora_moe_act = getattr(config_additional, 'LORA_MOE_ACT', 'linear')
            self.lora_moe_lambda = getattr(config_additional, 'LORA_MOE_LAMBDA', 1.0)
            self.lora_moe_softmax = getattr(config_additional, 'LORA_MOE_SOFTMAX', 0)
            self.lora_moe_group = getattr(config_additional, 'LORA_MOE_GROUP', 1)
            self.lora_moe = getattr(config_additional, 'LORA_MOE', 0)

            conf_lora_dropout = getattr(config_additional, 'LORA_DROPOUT', 0)
            self.lora_dropout = None
            if conf_lora_dropout > 0:
                self.lora_dropout = nn.Dropout(conf_lora_dropout)

            conf_lora_r_dropout = getattr(config_additional, 'LORA_R_DROPOUT', 0)
            self.lora_r_dropout = None
            if conf_lora_r_dropout > 0:
                self.lora_r_dropout = nn.Dropout(conf_lora_r_dropout)

            if self.lora_local_visual:
                self.conv_q_proj_adapter1 = self.build_conv_adapter(self.embed_dim, self.embed_dim, self.kernel_size,
                                                                    self.padding_q, self.stride_q)
            else:
                self.conv_q_proj_adapter1 = None
            self.q_proj_adapter1 = nn.Linear(self.embed_dim, self.lora_attn_dim, bias=False)
            nn.init.normal_(self.q_proj_adapter1.weight, std=0.02)
            self.q_proj_adapter2 = nn.Linear(self.lora_attn_dim, self.embed_dim, bias=False)
            self.q_proj_adapter2.weight.data.zero_()

            if self.lora_local_visual:
                self.conv_v_proj_adapter1 = self.build_conv_adapter(self.embed_dim, self.embed_dim, self.kernel_size,
                                                                    self.padding_kv, self.stride_kv)
            else:
                self.conv_v_proj_adapter1 = None
            self.v_proj_adapter1 = nn.Linear(self.embed_dim, self.lora_attn_dim, bias=False)
            nn.init.normal_(self.v_proj_adapter1.weight, std=0.02)
            self.v_proj_adapter2 = nn.Linear(self.lora_attn_dim, self.embed_dim, bias=False)
            self.v_proj_adapter2.weight.data.zero_()

            if self.lora_add_where == 'v1' or self.lora_add_where == 'v2':
                if self.lora_local_visual:
                    self.conv_k_proj_adapter1 = self.build_conv_adapter(self.embed_dim, self.embed_dim,
                                                                        self.kernel_size, self.padding_kv,
                                                                        self.stride_kv)
                else:
                    self.conv_k_proj_adapter1 = None
                self.k_proj_adapter1 = nn.Linear(self.embed_dim, self.lora_attn_dim, bias=False)
                nn.init.normal_(self.k_proj_adapter1.weight, std=0.02)
                self.k_proj_adapter2 = nn.Linear(self.lora_attn_dim, self.embed_dim, bias=False)
                self.k_proj_adapter2.weight.data.zero_()
                self.k_moe_adapter1 = None
                if self.lora_add_where == 'v2':
                    if self.lora_local_visual:
                        raise NotImplementedError('not implemented conv adapter for fc')
                    self.fc_proj_adapter1 = nn.Linear(self.embed_dim, self.lora_attn_dim, bias=False)
                    nn.init.normal_(self.fc_proj_adapter1.weight, std=0.02)
                    self.fc_proj_adapter2 = nn.Linear(self.lora_attn_dim, self.embed_dim, bias=False)
                    self.fc_proj_adapter2.weight.data.zero_()
                    self.fc_moe_adapter1 = None

            self.q_moe_adapter1 = None
            self.v_moe_adapter1 = None

            if self.lora_moe == 1:
                num_expert = self.lora_attn_dim // self.lora_moe_group

                self.q_moe_adapter1 = nn.Linear(self.embed_dim, num_expert, bias=False)
                nn.init.normal_(self.q_moe_adapter1.weight, std=0.02)

                self.v_moe_adapter1 = nn.Linear(self.embed_dim, num_expert, bias=False)
                nn.init.normal_(self.v_moe_adapter1.weight, std=0.02)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def convit_local_init(self):
        locality_strength = self.convit_local_strength
        # self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.convit_pos_proj.weight.data[position, 2] = -1
                self.convit_pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.convit_pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.convit_pos_proj.weight.data *= locality_strength

    def build_conv_adapter(self, dim_in, dim_out, kernel_size, padding, stride):
        proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=dim_in
            )),
            ('bn', nn.BatchNorm2d(dim_in)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj

    def build_cvt_dw(self, embed_dim):
        proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=self.cvt_v_kernel,
                padding=self.cvt_v_pad,
                stride=self.cvt_v_stride,
                bias=False,
                groups=embed_dim
            )),
            ('bn', nn.BatchNorm2d(embed_dim)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj

    def convit_forward(self, attn_output_weights, tgt_len, bsz, embed_dim, h, w):
        # attn_output_weights: [B*H, length, length]
        N = tgt_len - 1
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)
        pos_score = self.rel_indices.expand(bsz, -1, -1, -1)  # (B, total-1, total-1, 3)
        pos_score = self.convit_pos_proj(pos_score).permute(0, 3, 1, 2)  # (B, H, total-1, total-1)
        pos_score = pos_score.softmax(dim=-1)
        gating = self.convit_gating_param.view(1, -1, 1, 1)
        gating = gating.repeat(bsz, 1, 1, 1).view(-1, 1, 1)

        # Pad position score of cls token
        cls_pad_tensor = pos_score.new_full([pos_score.size(0), pos_score.size(1), 1, N], 0)
        pos_score = torch.cat([cls_pad_tensor, pos_score], dim=-2)
        cls_pad_tensor = pos_score.new_full([pos_score.size(0), pos_score.size(1), N + 1, 1], 0)
        pos_score = torch.cat([cls_pad_tensor, pos_score], dim=-1)
        pos_score = pos_score.view(-1, pos_score.size(-2), pos_score.size(-1))

        attn = (1. - torch.sigmoid(gating)) * attn_output_weights + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)

        return attn

    def container_forward(self, attn_output, value, tgt_len, bsz, embed_dim, h, w):
        # attn_output: [tgt_len, bsz, embed_dim]
        # value:[bsz * self.num_heads, src_len, self.head_dim]
        N = tgt_len - 1
        assert N == h * w
        cls_token_fea = torch.split(attn_output, [1, h * w], 0)[0]  # [1, bsz, embed_dim]
        cls_token_fea = cls_token_fea.transpose(0, 1)  # [bsz, 1, embed_dim]
        conv_value = value.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim).transpose(0,
                                                                                                1)  # [bsz, tgt_len, embed]
        _, conv_value = torch.split(conv_value, [1, h * w], 1)
        conv_value = rearrange(conv_value, 'b (h w) c -> b c h w', h=h, w=w)
        conv_out = self.container_conv(conv_value)
        conv_out = torch.cat((cls_token_fea, conv_out), dim=1)  # B, 1+HW, C
        conv_out = conv_out.permute(1, 0, 2)  # tgt_len, B, C
        mix_output = torch.sigmoid(self.container_gating) * attn_output + (
                    1 - torch.sigmoid(self.container_gating)) * conv_out

        return mix_output

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.convit_pos_proj.weight.device
        self.rel_indices = rel_indices.to(device)

    def adapter_forward(self, x, weight_1, weight_2, g_weight=None, conv_flag=False, conv_proj=None, h=None, w=None):
        if conv_flag:
            x = x.permute(1, 0, 2)  # (HW+1, B, C) -> (B, HW+1, C)
            cls_token, x = torch.split(x, [1, h * w], 1)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            conv_x = conv_proj(x)
            x = torch.cat((cls_token, conv_x), dim=1)  # B, HW+1, C
            x = x.permute(1, 0, 2)  # (B, HW+1, C) -> (HW+1, B, C)

        scale_factor = self.lora_attn_alpha / self.lora_attn_dim
        result = torch.matmul(x, weight_1.type_as(x).T)

        if self.lora_r_dropout is not None:
            result = self.lora_r_dropout(result)

        if g_weight is not None:
            g = torch.matmul(x, g_weight.weight.type_as(x).T)
            if self.lora_moe_act == 'sigmoid':
                g = torch.sigmoid(g)
            elif self.lora_moe_act == 'tanh':
                g = torch.tanh(g)
            elif self.lora_moe_act == 'relu':
                g = torch.relu(g)

            g = g * self.lora_moe_lambda

            if self.lora_moe_softmax == 1:
                g = torch.softmax(g, dim=-1)

            result = result.view(result.shape[0], result.shape[1], result.shape[2] // self.lora_moe_group,
                                 self.lora_moe_group) * g.unsqueeze(-1)
            result = result.view(result.shape[0], result.shape[1], -1)

        return torch.matmul(result, weight_2.type_as(x).T) * scale_factor

    def cvt_dw_forward(self, x, cvt_dw, h, w):
        x = x.permute(1, 0, 2)  # (HW+1, B, C) -> (B, HW+1, C)
        cls_token, x = torch.split(x, [1, h * w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        conv_x = cvt_dw(x)
        x = torch.cat((cls_token, conv_x), dim=1)  # B, HW+1, C
        x = x.permute(1, 0, 2)  # (B, HW+1, C) -> (HW+1, B, C)
        return x

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, h=None, w=None):
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        scaling = float(self.head_dim) ** -0.5
        if self.cvt_flag:
            assert self.cvt_q is not None or self.cvt_k is not None or self.cvt_v is not None
            if self.cvt_q is not None:
                query = self.cvt_dw_forward(query, self.cvt_q, h, w)
            if self.cvt_k is not None:
                key = self.cvt_dw_forward(key, self.cvt_k, h, w)
            if self.cvt_v is not None:
                value = self.cvt_dw_forward(value, self.cvt_v, h, w)
                #
        # compute in-projection
        #
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)

        if self.lora_attn_dim > 0:
            # value += self.adapter_forward(hidden_states, self.v_proj_adapter1.weight, self.v_proj_adapter2.weight)

            lora_input = query
            if self.lora_dropout is not None:
                lora_input = self.lora_dropout(lora_input)

            query_delta = self.adapter_forward(lora_input, self.q_proj_adapter1.weight, self.q_proj_adapter2.weight,
                                               g_weight=self.q_moe_adapter1, conv_flag=self.lora_local_visual,
                                               conv_proj=self.conv_q_proj_adapter1, h=h, w=w)
            value_delta = self.adapter_forward(lora_input, self.v_proj_adapter1.weight, self.v_proj_adapter2.weight,
                                               g_weight=self.v_moe_adapter1, conv_flag=self.lora_local_visual,
                                               conv_proj=self.conv_v_proj_adapter1, h=h, w=w)
            if self.lora_add_where == 'v1' or self.lora_add_where == 'v2':
                key_delta = self.adapter_forward(lora_input, self.k_proj_adapter1.weight, self.k_proj_adapter2.weight,
                                                 g_weight=self.k_moe_adapter1, conv_flag=self.lora_local_visual,
                                                 conv_proj=self.conv_k_proj_adapter1, h=h, w=w)
                k = k.contiguous() + key_delta

            q = q.contiguous() + query_delta
            v = v.contiguous() + value_delta

        # prep attention mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # scale the Q
        q = q * scaling
        # reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if self.output_raw_attention_weight:
            raw_attn_output_weights = attn_output_weights.data.cpu()

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        # (deep breath) calculate attention and out projection
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)

        if self.convit_v_flag:
            attn_output_weights = self.convit_forward(attn_output_weights, tgt_len, bsz, embed_dim, h, w)

        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.container_v_flag:
            attn_output = self.container_forward(attn_output, v, tgt_len, bsz, embed_dim, h, w)

        if self.lora_add_where == 'v2' and self.lora_attn_dim > 0:
            fc_delta = self.adapter_forward(attn_output, self.fc_proj_adapter1.weight, self.fc_proj_adapter2.weight,
                                            g_weight=self.fc_moe_adapter1)

        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.lora_add_where == 'v2' and self.lora_attn_dim > 0:
            attn_output = attn_output.contiguous() + fc_delta

        if self.output_raw_attention_weight:
            return attn_output, raw_attn_output_weights

        return attn_output, attn_output_weights


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 gumbel_select=False,
                 gumbel_addtwo=False,
                 custom_config=None,
                 modality=None,
                 convit_layer_flag=False,
                 cvt_layer_flag=False,
                 adapter_layer_flag=False,
                 ):
        super().__init__()

        self.modality = modality
        self.custom_attn = getattr(custom_config, 'CUSTOM_ATTN', False)
        # >>>>>>>>>>> For NMI output feature <<<<<<<<<<<<
        self.output_before_attn = getattr(custom_config, 'OUTPUT_BEFORE_ATTN', False)
        self.output_after_attn = getattr(custom_config, 'OUTPUT_AFTER_ATTN', False)
        self.output_after_attn_ln = getattr(custom_config, 'OUTPUT_AFTER_ATTN_LN', False)

        # >>>>>>>>>>> For CVT <<<<<<<<<<<<<
        self.cvt_flag_outside = getattr(custom_config, 'CVT_IN_V', False) and cvt_layer_flag and not getattr(
            custom_config, 'CVT_INSIDE', False)
        self.cvt_flag_inside = getattr(custom_config, 'CVT_IN_V', False) and cvt_layer_flag and getattr(custom_config,
                                                                                                        'CVT_INSIDE',
                                                                                                        False)

        if getattr(custom_config, 'LORA_OPEN', False) or getattr(custom_config, 'CUSTOM_ATTN', False):
            self.attn = Attention_CUST(d_model, n_head, config_additional=custom_config, modality=modality,
                                       convit_layer_flag=convit_layer_flag, cvt_flag=self.cvt_flag_inside)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gumbel_select = gumbel_select
        self.gumbel_addtwo = gumbel_addtwo
        if gumbel_select:
            # self.gumbel_logit = nn.Parameter(torch.randn(1, 2))
            if getattr(custom_config, 'LORA_OPEN', False) or getattr(custom_config, 'CUSTOM_ATTN', False):
                self.specific_attn = Attention_CUST(d_model, n_head, config_additional=custom_config, modality=modality)
            else:
                self.specific_attn = nn.MultiheadAttention(d_model, n_head)
            self.specific_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
            self.specific_ln_1 = LayerNorm(d_model)
            self.specific_ln_2 = LayerNorm(d_model)

        # >>>>>>>>>>> For ADAPTER <<<<<<<<<<<<<
        self.adapter_flag = getattr(custom_config, 'ADAPTER_FLAG', False) and adapter_layer_flag
        self.adapter_dim = getattr(custom_config, 'ADAPTER_ATTN_DIM', 0)
        if self.adapter_flag:
            assert self.adapter_dim > 0
            self.adapter_attn = nn.Sequential(OrderedDict([
                ("down_proj", nn.Linear(d_model, self.adapter_dim)),
                ("gelu", QuickGELU()),
                ("up_proj", nn.Linear(self.adapter_dim, d_model))
            ]))
            self.adapter_ffn = nn.Sequential(OrderedDict([
                ("down_proj", nn.Linear(d_model, self.adapter_dim)),
                ("gelu", QuickGELU()),
                ("up_proj", nn.Linear(self.adapter_dim, d_model))
            ]))

        # >>>>>>>>>>> For CVT <<<<<<<<<<<<<
        # self.cvt_flag = getattr(custom_config, 'CVT_IN_V', False) and cvt_layer_flag
        self.cvt_v_kernel = getattr(custom_config, 'CVT_V_KERNEL', False)
        self.cvt_v_stride = getattr(custom_config, 'CVT_V_STRIDE', False)
        self.cvt_v_pad = getattr(custom_config, 'CVT_V_PAD', False)
        self.cvt_res = getattr(custom_config, 'CVT_V_RES', False)

        if self.cvt_flag_outside:
            assert self.modality == 'visual'
            if getattr(custom_config, 'THREE_DWC_IN_CVT', False):
                self.cvt_dw = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn_1', nn.BatchNorm2d(d_model)),
                    ('conv_2', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn_2', nn.BatchNorm2d(d_model)),
                    ('conv_3', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn_3', nn.BatchNorm2d(d_model)),
                    ('rearrage', Rearrange('b c h w -> b (h w) c')),
                ]))
            elif getattr(custom_config, 'TWO_DWC_IN_CVT', False):
                self.cvt_dw = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn_1', nn.BatchNorm2d(d_model)),
                    ('conv_2', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn_2', nn.BatchNorm2d(d_model)),
                    ('rearrage', Rearrange('b c h w -> b (h w) c')),
                ]))
            else:
                self.cvt_dw = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=self.cvt_v_kernel,
                        padding=self.cvt_v_pad,
                        stride=self.cvt_v_stride,
                        bias=False,
                        groups=d_model
                    )),
                    ('bn', nn.BatchNorm2d(d_model)),
                    ('rearrage', Rearrange('b c h w -> b (h w) c')),
                ]))
            if self.cvt_res:
                self.ln_cvt = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, h=None, w=None, output_last_attnmap=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        if self.custom_attn:
            if output_last_attnmap:
                output = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, h=h, w=w)
                return output[0], output[1]
            else:
                return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, h=h, w=w)[0]
        else:
            if output_last_attnmap:
                raise NotImplementedError('not implemented for output_last_attnmap in original attn')
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_specific(self, x: torch.Tensor, h=None, w=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        if self.custom_attn:
            return self.specific_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, h=h, w=w)[0]
        else:
            return self.specific_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, action=None, h=None, w=None, output_last_attnmap=False, output_layer_fea=False):
        assert (self.gumbel_select and action is not None) or (not self.gumbel_select and action is None)
        if self.gumbel_select:
            assert not self.adapter_flag
            assert not self.cvt_flag_outside
            if self.gumbel_addtwo:
                x1 = x + self.drop_path(self.attention(self.ln_1(x), h=h, w=w))
                x1 = x1 + self.drop_path(self.mlp(self.ln_2(x1)))

                x2 = x + self.drop_path(self.attention_specific(self.specific_ln_1(x), h=h, w=w))
                x2 = x2 + self.drop_path(self.specific_mlp(self.specific_ln_2(x2)))
                assert action[0] + action[1] == 1
                x = x1 * action[0] + x2 * action[1]

            else:
                # >>>>>>>> Version 1 <<<<<<<<<<<<<
                if action.argmax(-1) == 0:
                    # for name, param in self.attn.named_parameters():
                    #     param.requires_grad = True
                    # for name, param in self.mlp.named_parameters():
                    #     param.requires_grad = True

                    assert action[0] == 1
                    x1 = x + self.drop_path(self.attention(self.ln_1(x), h=h, w=w))
                    x1 = x1 + self.drop_path(self.mlp(self.ln_2(x1)))
                    x = x1 * action[0]

                    # for name, param in self.specific_attn.named_parameters():
                    #     param.requires_grad = False
                    # for name, param in self.specific_mlp.named_parameters():
                    #     param.requires_grad = False

                elif action.argmax(-1) == 1:
                    # for name, param in self.specific_attn.named_parameters():
                    #     param.requires_grad = True
                    # for name, param in self.specific_mlp.named_parameters():
                    #     param.requires_grad = True
                    assert action[1] == 1
                    x2 = x + self.drop_path(self.attention_specific(self.specific_ln_1(x), h=h, w=w))
                    x2 = x2 + self.drop_path(self.specific_mlp(self.specific_ln_2(x2)))
                    x = x2 * action[1]

                    # for name, param in self.attn.named_parameters():
                    #     param.requires_grad = False
                    # for name, param in self.mlp.named_parameters():
                    #     param.requires_grad = False

        else:
            if self.adapter_flag:
                x = x + self.drop_path(self.adapter_attn(self.attention(self.ln_1(x), h=h, w=w)))
                x = x + self.drop_path(self.adapter_ffn(self.mlp(self.ln_2(x))))
            elif self.cvt_flag_outside:
                if self.modality == 'visual':
                    # >>>>>>>>>>>>> Insert Depth-wise Conv before QAV <<<<<<<<<<<<<<<<
                    if self.cvt_res:
                        x = x.permute(1, 0, 2)  # (HW+1, B, C) -> (B, HW+1, C)
                        cls_token, conv_x = torch.split(x, [1, h * w], 1)
                        conv_x = rearrange(conv_x, 'b (h w) c -> b c h w', h=h, w=w)
                        conv_x = self.cvt_dw(conv_x)  # B, HW, C
                        conv_x = torch.cat((cls_token, conv_x), dim=1)  # B, HW+1, C
                        x = self.ln_cvt(x.permute(1, 0, 2) + conv_x.permute(1, 0, 2))  # (B, HW+1, C) -> (HW+1, B, C)
                    else:
                        x = x.permute(1, 0, 2)  # (HW+1, B, C) -> (B, HW+1, C)
                        cls_token, x = torch.split(x, [1, h * w], 1)
                        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
                        conv_x = self.cvt_dw(x)
                        x = torch.cat((cls_token, conv_x), dim=1)  # B, HW+1, C
                        x = x.permute(1, 0, 2)  # (B, HW+1, C) -> (HW+1, B, C)

                x = x + self.drop_path(self.attention(self.ln_1(x), h=h, w=w))
                x = x + self.drop_path(self.mlp(self.ln_2(x)))
            else:
                if output_last_attnmap:
                    attn_output, last_attnmap = self.attention(self.ln_1(x), h=h, w=w, output_last_attnmap=True)
                    x = x + self.drop_path(attn_output)
                elif output_layer_fea:
                    if self.output_before_attn:
                        layer_output_fea = self.ln_1(x)
                        x = x + self.drop_path(self.attention(layer_output_fea, h=h, w=w))
                    elif self.output_after_attn:
                        layer_output_fea = self.attention(self.ln_1(x), h=h, w=w)
                        x = x + self.drop_path(layer_output_fea)
                    elif self.output_after_attn_ln:
                        x = x + self.drop_path(self.attention(self.ln_1(x), h=h, w=w))
                        layer_output_fea = self.ln_2(x)
                else:
                    x = x + self.drop_path(self.attention(self.ln_1(x), h=h, w=w))
                x = x + self.drop_path(self.mlp(self.ln_2(x)))
        if output_last_attnmap:
            return x, last_attnmap
        elif output_layer_fea:
            return x, layer_output_fea
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 modality=None,
                 custom_config=None,
                 ):
        super().__init__()
        self.modality = modality
        self.custom_attn = getattr(custom_config, 'CUSTOM_ATTN', False)
        if getattr(custom_config, 'LORA_OPEN', False) or getattr(custom_config, 'CUSTOM_ATTN', False):
            self.attn = Attention_CUST(d_model, n_head, config_additional=custom_config, modality=modality)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_context = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key, value, h=None, w=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        if self.custom_attn:
            return self.attn(x, key, value, need_weights=False, attn_mask=self.attn_mask, h=h, w=w)[0]
        else:
            return self.attn(x, key, value, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, key, value, h=None, w=None):
        x = x + self.drop_path(
            self.attention(self.ln_1(x), key=self.norm_context(key), value=self.norm_context(value), h=h, w=w))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class CrossAttentionLayer_Diffdim(nn.Module):
    def __init__(self,
                 input_q_dim: int,
                 input_k_dim: int,
                 input_v_dim: int,
                 output_qk_dim: int,
                 output_v_dim: int,
                 head_dim: int,
                 dropout=0.,
                 bias=True,
                 add_linear=False,
                 custom_config=None,
                 ):
        super(CrossAttentionLayer_Diffdim, self).__init__()
        self.head_dim = head_dim
        assert output_qk_dim == output_qk_dim == output_qk_dim, 'not implemented qkv output different dim'
        assert output_qk_dim % head_dim == 0
        self.num_heads = output_qk_dim // head_dim
        self.dropout = dropout
        self.add_linear = add_linear
        self.bias = bias

        self.q_proj_weight = nn.Parameter(torch.empty(output_qk_dim, input_q_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(output_qk_dim, input_k_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(output_v_dim, input_v_dim))
        self.register_parameter('in_proj_weight', None)
        if bias:
            self.q_in_proj_bias = nn.Parameter(torch.empty(output_qk_dim))
            self.k_in_proj_bias = nn.Parameter(torch.empty(output_qk_dim))
            self.v_in_proj_bias = nn.Parameter(torch.empty(output_v_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if self.add_linear:
            self.out_proj = _LinearWithBias(output_v_dim, output_v_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.bias:
            constant_(self.q_in_proj_bias, 0.)
            constant_(self.k_in_proj_bias, 0.)
            constant_(self.v_in_proj_bias, 0.)
        if self.add_linear:
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = F.linear(query, self.q_proj_weight, self.q_in_proj_bias)
        k = F.linear(key, self.k_proj_weight, self.k_in_proj_bias)
        v = F.linear(value, self.v_proj_weight, self.v_in_proj_bias)
        scaling = float(self.head_dim) ** -0.5
        # prep attention mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # scale the Q
        q = q * scaling
        # reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        # (deep breath) calculate attention and out projection
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)

        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.add_linear:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output


class CrossAttentionLayer_Window(nn.Module):
    def __init__(self,
                 input_q_dim: int,
                 input_kv_dim: int,
                 output_qk_dim: int,
                 output_v_dim: int,
                 head_qk_dim: int,
                 head_v_dim: int,
                 window_size_q: int,
                 window_size_kv: int,
                 dwconv_kv=False,
                 dropout=0.,
                 bias=True,
                 add_linear=False,
                 output_dim=0,
                 slide_window=False,
                 slide_window_kernel=0,
                 slide_window_pad=0,
                 slide_window_stride=0,
                 custom_config=None,
                 ):
        super(CrossAttentionLayer_Window, self).__init__()
        self.head_qk_dim = head_qk_dim
        self.head_v_dim = head_v_dim
        assert output_qk_dim % head_qk_dim == 0
        assert output_v_dim % head_v_dim == 0
        assert output_qk_dim // head_qk_dim == output_v_dim // head_v_dim
        self.num_heads = output_qk_dim // head_qk_dim
        self.dropout = dropout
        self.add_linear = add_linear
        if output_dim == 0:
            output_dim = output_v_dim
        self.bias = bias
        self.dwconv_kv = dwconv_kv
        self.window_size_q = window_size_q
        self.window_size_kv = window_size_kv
        self.output_v_dim = output_v_dim
        self.output_qk_dim = output_qk_dim
        self.slide_window = slide_window
        if self.slide_window:
            self.slide_window_kernel = slide_window_kernel
            self.slide_window_pad = slide_window_pad
            self.slide_window_stride = slide_window_stride
            self.unfold = nn.Unfold(kernel_size=(slide_window_kernel, slide_window_kernel),
                                    stride=slide_window_stride, padding=slide_window_pad)
        self.relative_position_bias = getattr(custom_config, 'T2B_WINDOWATTN_RELATIVE_POS')
        if self.relative_position_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size_q + window_size_kv - 1) * (window_size_q + window_size_kv - 1),
                            self.num_heads))  # kh+qh-1 * kw+qw-1, nH

            # get pair-wise relative position index for each token inside the window
            kv_coords_h = torch.arange(window_size_kv)
            kv_coords_w = torch.arange(window_size_kv)
            kv_coords = torch.stack(torch.meshgrid([kv_coords_h, kv_coords_w]))  # 2, kh, kw
            kv_coords_flatten = torch.flatten(kv_coords, 1)  # 2, kh*kw

            q_coords_h = torch.arange(window_size_q)
            q_coords_w = torch.arange(window_size_q)
            q_coords = torch.stack(torch.meshgrid([q_coords_h, q_coords_w]))  # 2, qh, qw
            q_coords_flatten = torch.flatten(q_coords, 1)  # 2, qh*qw

            relative_coords = q_coords_flatten[:, :, None] - kv_coords_flatten[:, None, :]  # 2, qh*qw, kh*kw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # qh*qw, kh*kw, 2
            relative_coords[:, :, 0] += window_size_kv - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size_kv - 1
            relative_coords[:, :, 0] *= window_size_kv + window_size_q - 1
            relative_position_index = relative_coords.sum(-1)  # qh*qw, kh*kw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.q_proj_weight = nn.Parameter(torch.empty(output_qk_dim, input_q_dim))
        if dwconv_kv:
            self.k_dwconv = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    input_kv_dim,
                    input_kv_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                    groups=input_kv_dim
                )),
                ('bn', nn.BatchNorm2d(input_kv_dim)),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
            self.v_dwconv = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    input_kv_dim,
                    input_kv_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                    groups=input_kv_dim
                )),
                ('bn', nn.BatchNorm2d(input_kv_dim)),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        else:
            self.k_proj_weight = nn.Parameter(torch.empty(output_qk_dim, input_kv_dim))
            self.v_proj_weight = nn.Parameter(torch.empty(output_v_dim, input_kv_dim))
        self.register_parameter('in_proj_weight', None)
        if bias:
            self.q_in_proj_bias = nn.Parameter(torch.empty(output_qk_dim))
            if not dwconv_kv:
                self.k_in_proj_bias = nn.Parameter(torch.empty(output_qk_dim))
                self.v_in_proj_bias = nn.Parameter(torch.empty(output_v_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if self.add_linear:
            self.out_proj = _LinearWithBias(output_v_dim, output_dim)

        self.bottom_dw_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                input_q_dim,
                input_q_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
                groups=input_q_dim
            )),
            ('bn', nn.BatchNorm2d(input_q_dim)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
            # TODO: add projection
        ]))
        self.ln_adapt = LayerNorm(output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        if not self.dwconv_kv:
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.bias:
            constant_(self.q_in_proj_bias, 0.)
            if not self.dwconv_kv:
                constant_(self.k_in_proj_bias, 0.)
                constant_(self.v_in_proj_bias, 0.)
        if self.add_linear:
            constant_(self.out_proj.bias, 0.)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W, n_head):
        """
        Args:
            windows: (num_windows*B*n_head, window_size*window_size, C_head)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        # (nw*B*n_head, wz*wz, c_head) -> (nw*B, wz*wz, n_head, c_head)
        windows = windows.view(-1, n_head, windows.size(1), windows.size(2)).permute(0, 2, 1, 3)
        # (nw*B, wz*wz, n_head, c_head) -> (nw*B, wz, wz, n_head, c_head)
        windows = windows.view(windows.size(0), window_size, window_size, windows.size(-2), windows.size(-1))
        # (nw*B, wz, wz, n_head, c_head) -> (nw*B, wz, wz, c_output)
        windows = windows.view(windows.size(0), windows.size(1), windows.size(2), n_head * windows.size(-1))

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def cross_attn(self, query, key, value, attn_mask):
        # Below computing cross-attention from top to bottom
        bsz, input_q_dim, q_h, q_w = query.shape
        bsz, input_kv_dim, kv_h, kv_w = key.shape
        assert key.shape == value.shape

        query_flat = rearrange(query, 'b c h w -> (h w) b c', h=q_h, w=q_w)
        q = F.linear(query_flat, self.q_proj_weight, self.q_in_proj_bias)
        q = rearrange(q, '(h w) b c -> b h w c', h=q_h, w=q_w)

        if self.dwconv_kv:
            k = self.k_dwconv(key).permute(0, 2, 3, 1)  # b,c,h,w -> b,h,w,c
            v = self.v_dwconv(value).permute(0, 2, 3, 1)  # b,c,h,w -> b,h,w,c
        else:
            key_flat = rearrange(key, 'b c h w -> (h w) b c', h=kv_h, w=kv_w)
            value_flat = rearrange(value, 'b c h w -> (h w) b c', h=kv_h, w=kv_w)
            k = F.linear(key_flat, self.k_proj_weight, self.k_in_proj_bias)
            v = F.linear(value_flat, self.v_proj_weight, self.v_in_proj_bias)
            k = rearrange(k, '(h w) b c -> b h w c', h=kv_h, w=kv_w)
            v = rearrange(v, '(h w) b c -> b h w c', h=kv_h, w=kv_w)

        scaling = float(self.head_qk_dim) ** -0.5
        # prep attention mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # scale the Q
        q = q * scaling

        if self.slide_window:
            qk_window_num = (q_w // self.slide_window_stride) * (q_h // self.slide_window_stride)
        else:
            qk_window_num = (q_w // self.window_size_q) * (q_h // self.window_size_q)
        # TODO: partition q,k,v according to the window size
        q_windows = self.window_partition(q, self.window_size_q)  # nW*B, window_size, window_size, C
        if self.slide_window:
            k_windows = self.unfold(
                k.permute(0, 3, 1, 2))  # (B, C, H, W) -> (B, C*window_size*window_size, n_window*n_window)
            k_windows = k_windows.view(k_windows.size(0), self.output_qk_dim, self.window_size_kv, self.window_size_kv,
                                       k_windows.size(-1))  # (B, C, window_size, window_size, n_window*n_window)
            k_windows = k_windows.transpose(1, -1).contiguous().view(-1, self.window_size_kv, self.window_size_kv,
                                                                     self.output_qk_dim)  # (B*n_window*n_window, window_size, window_size, C)

            v_windows = self.unfold(
                v.permute(0, 3, 1, 2))  # (B, C, H, W) -> (B, C*window_size*window_size, n_window*n_window)
            v_windows = v_windows.view(v_windows.size(0), self.output_v_dim, self.window_size_kv, self.window_size_kv,
                                       v_windows.size(-1))  # (B, C, window_size, window_size, n_window*n_window)
            v_windows = v_windows.transpose(1, -1).contiguous().view(-1, self.window_size_kv, self.window_size_kv,
                                                                     self.output_v_dim)  # (B*n_window*n_window, window_size, window_size, C)
            if self.slide_window_pad != 0:
                assert attn_mask is None
                attn_mask = torch.ones([1, 1, q_h, q_w], dtype=v_windows.dtype, device=v_windows.device)
                attn_mask = self.unfold(attn_mask)  # (1, 1, H, W) -> (1, window_size*window_size, n_window*n_window)
                window_size = attn_mask.size(1)
                attn_mask = attn_mask.repeat(bsz, 1, 1).transpose(1, 2).contiguous().view(-1, window_size).unsqueeze(
                    1)  # (b*nw, 1, wds*wds)
                attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).contiguous().view(-1, 1,
                                                                                                     attn_mask.size(
                                                                                                         -1))  # (b*nw*nh, 1, wds*wds)
                attn_mask[attn_mask == 0] = float('-inf')
                attn_mask[attn_mask == 1] = 0
        else:
            k_windows = self.window_partition(k, self.window_size_kv)  # nW*B, window_size, window_size, C
            v_windows = self.window_partition(v, self.window_size_kv)  # nW*B, window_size, window_size, C

        # reshape q, k, v for multihead attention and make em batch first
        # nw*B, wz, wz, c -> nw*B, wz*wz, c -> nw*B*nh, wz*wz, head_dim
        q_windows = q_windows.contiguous().view(q_windows.size(0), self.window_size_q * self.window_size_q,
                                                q_windows.size(-1))
        q_windows = q_windows.view(q_windows.size(0), q_windows.size(1), self.num_heads, self.head_qk_dim).permute(0, 2,
                                                                                                                   1,
                                                                                                                   3).contiguous()  # (nw*B, nh, wz*wz, head_dim)
        q = q_windows.view(q_windows.size(0) * self.num_heads, q_windows.size(2), self.head_qk_dim)

        k_windows = k_windows.contiguous().view(k_windows.size(0), self.window_size_kv * self.window_size_kv,
                                                k_windows.size(-1))
        k_windows = k_windows.view(k_windows.size(0), k_windows.size(1), self.num_heads, self.head_qk_dim).permute(0, 2,
                                                                                                                   1,
                                                                                                                   3).contiguous()  # (nw*B, nh, wz*wz, head_dim)
        k = k_windows.view(k_windows.size(0) * self.num_heads, k_windows.size(2), self.head_qk_dim)

        v_windows = v_windows.contiguous().view(v_windows.size(0), self.window_size_kv * self.window_size_kv,
                                                v_windows.size(-1))
        v_windows = v_windows.view(v_windows.size(0), v_windows.size(1), self.num_heads, self.head_v_dim).permute(0, 2,
                                                                                                                  1,
                                                                                                                  3).contiguous()  # (nw*B, nh, wz*wz, head_dim)
        v = v_windows.view(v_windows.size(0) * self.num_heads, v_windows.size(2), self.head_v_dim)

        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * (qk_window_num) * self.num_heads,
                                                    self.window_size_q * self.window_size_q,
                                                    self.window_size_kv * self.window_size_kv]

        if self.relative_position_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size_q * self.window_size_q, self.window_size_kv * self.window_size_kv,
                -1)  # qh*qw,kh*kw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, qh*qw,kh*kw
            # (bsz*nw*nh, qhw, khq) -> (bsz*nw, nh, qhw, khq) + (1, nh, qhw, khq)
            attn_output_weights = attn_output_weights.view(bsz * (qk_window_num), self.num_heads,
                                                           attn_output_weights.size(-2), attn_output_weights.size(-1)) \
                                  + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(bsz * (qk_window_num) * self.num_heads,
                                                           attn_output_weights.size(-2), attn_output_weights.size(-1))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        # (deep breath) calculate attention and out projection
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)

        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * (qk_window_num) * self.num_heads,
                                            self.window_size_q * self.window_size_q, self.head_v_dim]
        attn_output = self.window_reverse(attn_output, self.window_size_q, q_h, q_w, self.num_heads)  # (B, H, W, C)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(-1, bsz, self.output_v_dim)  # (H*W, B, C)
        # attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        return attn_output

    def forward(self, top_input, bottom_input, h, w, attn_mask=None):
        cls_token, bottom_grid_input = torch.split(bottom_input.permute(1, 0, 2), [1, h * w],
                                                   1)  # (HW+1, B, C) -> (B, HW+1, C)
        bottom_grid_input = rearrange(bottom_grid_input, 'b (h w) c -> b c h w', h=h, w=w)
        query = bottom_grid_input  # (b, c, h, w)
        key = top_input
        value = top_input

        # Compute bottom+dw conv -> bottom residual output
        bottom_output = self.bottom_dw_conv(bottom_grid_input.contiguous())
        bottom_output = torch.cat((cls_token, bottom_output), dim=1)

        attn_output = self.cross_attn(query, key, value, attn_mask)

        if self.add_linear:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        top2bottom = torch.cat([torch.zeros_like(cls_token), attn_output.transpose(0, 1).contiguous()],
                               dim=1)  # B, HW+1, C
        bottom_output = self.ln_adapt((bottom_output + top2bottom).permute(1, 0, 2))
        top_output = top_input

        return top_output, bottom_output


class Lateral_Adapter(nn.Module):
    def __init__(self,
                 top_dim: int,
                 bottom_dim: int,
                 top2bottom_kernel: int,
                 top2bottom_padding: int,
                 top2bottom_stride=None,
                 bottom2top_kernel=None,
                 bottom2top_padding=None,
                 bottom2top_stride=None,
                 custom_config=None,
                 bottom2top_last_layer=False,
                 bottom2top_layer_flag=False,
                 top2bottom_pool_kernel=0,
                 ):
        super().__init__()

        self.top2bottom_usecls = getattr(custom_config, 'PRALLEL_T2B_USECLS', False)
        self.top2bottom_bn_relu = getattr(custom_config, 'PRALLEL_T2B_ADD_BN_RELU', False)
        self.top2bottom_bn_ln_relu = getattr(custom_config, 'PRALLEL_T2B_ADD_BN_LN_RELU', False)
        self.top2bottom_noln_add = getattr(custom_config, 'PRALLEL_T2B_NOLN_ADD', False)

        if top2bottom_pool_kernel != 0:
            assert getattr(custom_config, 'PARALLEL_T2B_POOL_SIZE', False)
            self.top2bottom_pooling_flag = True
            self.sample_pooling = nn.AvgPool2d(kernel_size=top2bottom_pool_kernel, stride=top2bottom_pool_kernel)
        else:
            self.top2bottom_pooling_flag = False

        self.top2bottom_dw_conv = nn.Sequential()
        self.top2bottom_pw_conv = nn.Sequential()

        self.top2bottom_dw_conv.add_module('conv', nn.Conv2d(
            top_dim,
            top_dim,
            kernel_size=top2bottom_kernel,
            padding=top2bottom_padding,
            stride=top2bottom_stride,
            bias=False,
            groups=top_dim,
        ))

        self.top2bottom_pw_conv.add_module('conv', nn.Conv2d(
            top_dim,
            bottom_dim,
            kernel_size=1,
            bias=False,
        ))
        if self.top2bottom_bn_relu:
            self.top2bottom_dw_conv.add_module('bn', nn.BatchNorm2d(top_dim))
            self.top2bottom_dw_conv.add_module('relu', nn.ReLU(inplace=True))

            self.top2bottom_pw_conv.add_module('bn', nn.BatchNorm2d(bottom_dim))
            self.top2bottom_pw_conv.add_module('relu', nn.ReLU(inplace=True))
        elif self.top2bottom_bn_ln_relu:
            self.top2bottom_dw_conv.add_module('bn', nn.BatchNorm2d(top_dim))

            self.top2bottom_pw_conv.add_module('reshape1', Rearrange('b c h w -> b (h w) c'))
            self.top2bottom_pw_conv.add_module('ln', LayerNorm(bottom_dim))
            self.top2bottom_pw_conv.add_module('relu', nn.ReLU(inplace=True))
        else:
            self.top2bottom_dw_conv.add_module('bn', nn.BatchNorm2d(top_dim))

        # self.top2bottom_dw_conv = nn.Sequential(OrderedDict([
        #                             ('conv', nn.Conv2d(
        #                                 top_dim,
        #                                 top_dim,
        #                                 kernel_size=top2bottom_kernel,
        #                                 padding=top2bottom_padding,
        #                                 stride=top2bottom_stride,
        #                                 bias=False,
        #                                 groups=top_dim,
        #                             )),
        #                             ('bn', nn.BatchNorm2d(top_dim)),
        #                             # TODO: relu, check paper
        #                             ]))
        # self.top2bottom_pw_conv = nn.Conv2d(
        #                                 top_dim,
        #                                 bottom_dim,
        #                                 kernel_size=1,
        #                                 bias=False,
        #                             )
        self.bottom_dw_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                bottom_dim,
                bottom_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
                groups=bottom_dim
            )),
            ('bn', nn.BatchNorm2d(bottom_dim)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
            # TODO: add projection
        ]))

        if not self.top2bottom_noln_add:
            self.ln_adapt = LayerNorm(bottom_dim)

        self.bottom2top_flag = getattr(custom_config, 'PARALLEL_B2T', False) and bottom2top_layer_flag
        if bottom2top_last_layer and getattr(custom_config, 'PARALLEL_B2T_NO_LASTLAYER', False):
            self.bottom2top_flag = False
        self.bottom2top_bilinear = getattr(custom_config, 'PARALLEL_B2T_BILINEAR', False)
        self.bottom2top_crossattn = getattr(custom_config, 'PARALLEL_B2T_CROSSATTN', False)
        if self.bottom2top_flag:
            if self.bottom2top_bilinear:
                self.bottom2top_dw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        bottom_dim,
                        bottom_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=bottom_dim,
                    )),
                    ('bn', nn.BatchNorm2d(bottom_dim)),
                ]))
                self.bottom2top_pw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        bottom_dim,
                        top_dim,
                        kernel_size=1,
                        bias=False,
                    )),
                    ('bn', nn.BatchNorm2d(top_dim)),
                ]))
                self.top_dw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        top_dim,
                        top_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=top_dim
                    )),
                    ('bn', nn.BatchNorm2d(top_dim)),
                ]))
            elif self.bottom2top_crossattn:
                self.bottom2top_dw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        bottom_dim,
                        bottom_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=bottom_dim,
                    )),
                    ('bn', nn.BatchNorm2d(bottom_dim)),
                ]))
                self.top_dw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        top_dim,
                        top_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=top_dim
                    )),
                    ('bn', nn.BatchNorm2d(top_dim)),
                ]))
                self.bottom2top_crossattn = CrossAttentionLayer_Diffdim(input_q_dim=top_dim, input_k_dim=bottom_dim,
                                                                        input_v_dim=bottom_dim, \
                                                                        output_qk_dim=top_dim, output_v_dim=top_dim,
                                                                        head_dim=64, custom_config=custom_config)
                self.bottom2top_ln = LayerNorm(top_dim)
            else:
                self.bottom2top_dw_deconv = nn.Sequential(OrderedDict([
                    ('conv', nn.ConvTranspose2d(
                        bottom_dim,
                        bottom_dim,
                        kernel_size=bottom2top_kernel,
                        padding=bottom2top_padding,
                        stride=bottom2top_stride,
                        bias=False,
                        groups=bottom_dim,
                    )),
                    ('bn', nn.BatchNorm2d(bottom_dim)),
                ]))
                self.bottom2top_pw_deconv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        bottom_dim,
                        top_dim,
                        kernel_size=1,
                        bias=False,
                    )),
                    ('bn', nn.BatchNorm2d(top_dim)),
                ]))
                self.top_dw_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        top_dim,
                        top_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=top_dim
                    )),
                    ('bn', nn.BatchNorm2d(top_dim)),
                ]))

    def attention(self, x: torch.Tensor, key, value, h=None, w=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        if self.custom_attn:
            return self.attn(x, key, value, need_weights=False, attn_mask=self.attn_mask, h=h, w=w)[0]
        else:
            return self.attn(x, key, value, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, top_input, bottom_input, h, w):
        if self.top2bottom_pooling_flag:
            top2bottom = self.top2bottom_dw_conv(self.sample_pooling(top_input.contiguous()))
        else:
            top2bottom = self.top2bottom_dw_conv(top_input.contiguous())
        top2bottom = self.top2bottom_pw_conv(top2bottom)
        if self.top2bottom_bn_ln_relu:
            # shape is already [b, (hw), c]
            top2bottom = top2bottom
        else:
            top2bottom = rearrange(top2bottom, 'b c h w -> b (h w) c', h=h, w=w)

        bottom_input = bottom_input.permute(1, 0, 2)  # (HW+1, B, C) -> (B, HW+1, C)
        cls_token, bottom_grid_input = torch.split(bottom_input, [1, h * w], 1)
        bottom_grid_input = rearrange(bottom_grid_input, 'b (h w) c -> b c h w', h=h, w=w)
        bottom_output = self.bottom_dw_conv(bottom_grid_input.contiguous())
        bottom_output = torch.cat((cls_token, bottom_output), dim=1)  # B, HW+1, C

        if self.top2bottom_usecls:
            top2bottom = torch.cat([cls_token, top2bottom.contiguous()], dim=1)  # B, HW+1, C
        else:
            top2bottom = torch.cat([torch.zeros_like(cls_token), top2bottom.contiguous()], dim=1)  # B, HW+1, C
        if self.top2bottom_noln_add:
            bottom_output = (bottom_output + top2bottom).permute(1, 0, 2)
        else:
            bottom_output = self.ln_adapt((bottom_output + top2bottom).permute(1, 0, 2))
        top_output = top_input

        if self.bottom2top_flag:
            if self.bottom2top_bilinear:
                bottom2top = self.bottom2top_dw_conv(bottom_grid_input.contiguous())
                bottom2top = self.bottom2top_pw_conv(bottom2top)
                bottom2top = F.upsample(bottom2top, size=(top_output.size(2), top_output.size(3)), mode='bilinear',
                                        align_corners=False)
                top_output = self.top_dw_conv(top_input)
                top_output = top_output.contiguous() + bottom2top.contiguous()

            elif self.bottom2top_crossattn:
                bottom2top = self.bottom2top_dw_conv(bottom_grid_input.contiguous())
                top_output = self.top_dw_conv(top_input)
                top_output_flat = rearrange(top_output, 'b c h w -> b (h w) c', h=top_output.size(-2),
                                            w=top_output.size(-1))
                bottom2top_flat = rearrange(bottom2top, 'b c h w -> b (h w) c', h=h, w=w)
                bottom2top_flat = torch.cat((cls_token, bottom2top_flat), dim=1)
                top_output_attn = self.bottom2top_crossattn(top_output_flat.transpose(0, 1),
                                                            bottom2top_flat.transpose(0, 1),
                                                            bottom2top_flat.transpose(0, 1))
                top_output_flat = self.bottom2top_ln(top_output_flat.transpose(0, 1) + top_output_attn)  # (hw, b, c)
                top_output_flat = top_output_flat.transpose(0, 1)  # (hw, b, c)  -> (b, hw, c)
                top_output = rearrange(top_output_flat, 'b (h w) c -> b c h w', h=top_output.size(-2),
                                       w=top_output.size(-1))

            else:
                bottom2top = self.bottom2top_dw_deconv(bottom_grid_input.contiguous())
                bottom2top = self.bottom2top_pw_deconv(bottom2top)
                top_output = self.top_dw_conv(top_input)
                top_output = top_output.contiguous() + bottom2top.contiguous()
        return top_output, bottom_output


class ConvResBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 mid_dim: int,
                 output_dim: int,
                 kernel: int,
                 stride: int,
                 padding: int,
                 res_conv=False,
                 act_layer=nn.ReLU, ):
        super().__init__()
        # groups = 1
        self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dim, eps=1e-6)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel, stride=stride, groups=1, padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_dim, eps=1e-6)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(mid_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_dim, eps=1e-6)
        self.act3 = act_layer(inplace=True)

        self.res_conv = res_conv
        if res_conv:
            self.residual_conv = nn.Conv2d(in_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = nn.BatchNorm2d(output_dim, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)
        return x


class Resnet_Stage(nn.Module):
    def __init__(self,
                 num_layer: int,
                 in_dim: int,
                 mid_dim: int,
                 output_dim: int,
                 kernel: int,
                 stride: int,
                 padding: int, ):
        super().__init__()

        assert num_layer != 0
        self.resnet_stage = nn.Sequential()
        for i in range(num_layer):
            s = stride if i == 0 else 1
            in_channel = in_dim if i == 0 else output_dim
            res_conv = True if i == 0 else False
            self.resnet_stage.add_module('conv_' + str(i),
                                         ConvResBlock(
                                             in_dim=in_channel,
                                             mid_dim=mid_dim,
                                             output_dim=output_dim,
                                             kernel=kernel,
                                             stride=s,
                                             padding=padding,
                                             res_conv=res_conv,
                                         )
                                         )

    def forward(self, x):
        x = self.resnet_stage(x)
        return x


class ResBasicBlock_v0(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_dim: int,
            output_dim: int,
            stride: int = 2,
    ):
        super(ResBasicBlock_v0, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_dim, output_dim, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_dim)
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EarlyconvRes(nn.Module):
    def __init__(self,
                 in_dim: int,
                 output_dim: int,
                 first_conv_k: int,
                 res_block: str,
                 res_layers,
                 res_layer_strides,
                 ):
        super().__init__()

        assert in_dim == 3
        if first_conv_k == 3:
            self.conv1 = nn.Conv2d(3, output_dim // 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        elif first_conv_k == 7:
            self.conv1 = nn.Conv2d(3, output_dim // 16, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError

        self.bn1 = nn.BatchNorm2d(output_dim // 16)
        self.relu = nn.ReLU(inplace=True)

        self.resnet_stage = nn.Sequential()
        assert len(res_layers) == len(res_layer_strides)
        for i, layer_num_i in enumerate(res_layers):
            in_channel = output_dim // pow(2, len(res_layers) - i)
            out_channel = in_channel * 2
            if res_block == 'basic_v0':
                assert layer_num_i == 1
                self.resnet_stage.add_module('conv_' + str(i),
                                             ResBasicBlock_v0(
                                                 in_dim=in_channel,
                                                 output_dim=out_channel,
                                                 stride=res_layer_strides[i],
                                             )
                                             )
                # self.resnet_stage.add_module('conv_' + str(i),
                #     ResBasicBlock_v0(
                #         in_dim=in_channel,
                #         output_dim=out_channel,
                #         stride=2,
                #         )
                #     )
        self.last_conv = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resnet_stage(x)
        x = self.last_conv(x)
        # pdb.set_trace()
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 gumbel_select=False,
                 gumbel_addtwo=False,
                 custom_config=None,
                 modality=None,
                 first_conv=False
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.gumbel_select = gumbel_select
        self.first_conv = first_conv
        self.convit_v_flag = getattr(custom_config, 'CONVIT_IN_V', False) and (modality == 'visual')
        self.cvt_v_flag = getattr(custom_config, 'CVT_IN_V', False) and (modality == 'visual')
        self.adapter_flag = getattr(custom_config, 'ADAPTER_FLAG', False)
        self.perceiver_v_flag = getattr(custom_config, 'PERCEIVER_IN_V', False) and (modality == 'visual')
        self.perceiver_t_flag = getattr(custom_config, 'PERCEIVER_IN_T', False) and (modality == 'text')
        self.parallel_in_v = getattr(custom_config, 'PARALLEL_IN_V', False) and (modality == 'visual')

        if first_conv:
            assert getattr(custom_config, 'EARLY_CONV', False) and modality == 'visual'
            if self.cvt_v_flag:
                self.cvt_layers = getattr(custom_config, 'CVT_LAYERS', [])
                if self.cvt_layers == [0, ]:
                    raise NotImplementedError('layer 0 is set to be early conv')
                if len(self.cvt_layers) == 0:
                    self.cvt_layers = [True] * layers
                else:
                    self.cvt_layers = [True if ii in self.cvt_layers else False for ii in range(layers)]
                self.cvt_layers[0] = False
            else:
                self.cvt_layers = [False] * layers

            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, gumbel_select=gumbel_select,
                                           gumbel_addtwo=gumbel_addtwo, custom_config=custom_config, modality=modality,
                                           cvt_layer_flag=self.cvt_layers[_])
                    if _ >= 1 else
                    self.build_early_conv_block(width, custom_config=custom_config)
                    for _ in range(layers)
                ]
            )

        elif self.adapter_flag:
            self.adapter_layers = getattr(custom_config, 'ADAPTER_LAYERS', [])
            if len(self.adapter_layers) == 0:
                self.adapter_layers = [True] * layers
            else:
                self.adapter_layers = [True if ii in self.adapter_layers else False for ii in range(layers)]
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, gumbel_select=gumbel_select,
                                           gumbel_addtwo=gumbel_addtwo, custom_config=custom_config, modality=modality,
                                           adapter_layer_flag=self.adapter_layers[_])
                    for _ in range(layers)
                ]
            )

        elif self.cvt_v_flag:
            self.cvt_layers = getattr(custom_config, 'CVT_LAYERS', [])
            if len(self.cvt_layers) == 0:
                self.cvt_layers = [True] * layers
            else:
                self.cvt_layers = [True if ii in self.cvt_layers else False for ii in range(layers)]
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, gumbel_select=gumbel_select,
                                           gumbel_addtwo=gumbel_addtwo, custom_config=custom_config, modality=modality,
                                           cvt_layer_flag=self.cvt_layers[_])
                    for _ in range(layers)
                ]
            )
        elif self.convit_v_flag:
            self.convit_layers = getattr(custom_config, 'CONVIT_LAYERS', [])
            if len(self.convit_layers) == 0:
                self.convit_layers = [True] * layers
            else:
                self.convit_layers = [True if ii in self.convit_layers else False for ii in range(layers)]
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, gumbel_select=gumbel_select,
                                           gumbel_addtwo=gumbel_addtwo, custom_config=custom_config, modality=modality,
                                           convit_layer_flag=self.convit_layers[_])
                    for _ in range(layers)
                ]
            )
        else:
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, gumbel_select=gumbel_select,
                                           gumbel_addtwo=gumbel_addtwo, custom_config=custom_config, modality=modality)
                    for _ in range(layers)
                ]
            )

        if self.perceiver_v_flag or self.perceiver_t_flag:
            self.perceiver_layer = getattr(custom_config, 'PERCEIVER_LAYER', [])
            self.perceiver_num_latents = getattr(custom_config, 'PERCEIVER_N_LATENT', 0)
            self.perceiver_latent = nn.Parameter(torch.randn(self.perceiver_num_latents, width))
            if self.perceiver_t_flag:
                self.perceiver_latent2ori_blocks = nn.Sequential(
                    *[
                        ResidualCrossAttentionBlock(width, heads, None, drop_path, custom_config=custom_config,
                                                    modality=modality)
                        for _ in self.perceiver_layer
                    ]
                )
            else:
                self.perceiver_latent2ori_blocks = nn.Sequential(
                    *[
                        ResidualCrossAttentionBlock(width, heads, attn_mask, drop_path, custom_config=custom_config,
                                                    modality=modality)
                        for _ in self.perceiver_layer
                    ]
                )

        self.parallel_reuse_earlyconv_firstlayer = getattr(custom_config, 'PARALLEL_REUSE_EARLYCONV_FIRSTLAYER', False)
        self.parallel_reuse_earlyconv_alllayer = getattr(custom_config, 'PARALLEL_REUSE_EARLYCONV_ALLLAYER', False)
        if self.parallel_in_v:
            self.parallel_number_layers = getattr(custom_config, 'PARALLEL_N_LAYERS', 0)
            self.parallel_lateral_layers = getattr(custom_config, 'PARALLEL_LATERAL_LAYER', [])
            self.parallel_input_dims = [3, width // 16, width // 8, width // 4, width // 2]
            self.parallel_output_dims = [width // 16, width // 8, width // 4, width // 2, width]
            self.parallel_kernels = getattr(custom_config, 'PARALLEL_KERNELS', [3, 3, 3, 3, 3])
            self.parallel_paddings = getattr(custom_config, 'PARALLEL_PADDINGS', [1, 1, 1, 1, 1])
            self.parallel_strides = getattr(custom_config, 'PARALLEL_STRIDES', [2, 2, 2, 2, 2])
            if self.parallel_reuse_earlyconv_firstlayer or self.parallel_reuse_earlyconv_alllayer:
                self.parallel_input_dims[0] = width // 16
                self.parallel_strides[0] = 1
            if getattr(custom_config, 'PARALLEL_RESNET', False):
                expansion = 2
                self.number_res_layers = getattr(custom_config, 'PARALLEL_RESNET_LAYERS', False)
                self.parallel_branch_v = nn.Sequential(
                    *[
                        Resnet_Stage(
                            num_layer=self.number_res_layers[iii],
                            in_dim=self.parallel_input_dims[iii],
                            mid_dim=(self.parallel_output_dims[iii]) // expansion,
                            output_dim=self.parallel_output_dims[iii],
                            kernel=self.parallel_kernels[iii],
                            stride=self.parallel_strides[iii],
                            padding=self.parallel_paddings[iii],
                        )
                        if iii != 0 else
                        self.build_conv_bn_relu(self.parallel_input_dims[iii], self.parallel_output_dims[iii],
                                                self.parallel_kernels[iii], self.parallel_paddings[iii],
                                                self.parallel_strides[iii])
                        for iii in range(self.parallel_number_layers)
                    ]
                )
            else:
                self.parallel_branch_v = nn.Sequential(
                    *[
                        self.build_conv_bn_relu(self.parallel_input_dims[iii], self.parallel_output_dims[iii],
                                                self.parallel_kernels[iii], self.parallel_paddings[iii],
                                                self.parallel_strides[iii])
                        for iii in range(self.parallel_number_layers)
                    ]
                )
            self.specific2share_lateral_kernels = [16 + 2, 8 + 2, 4 + 2, 2 + 2, 1 + 2]
            self.specific2share_lateral_paddings = [1, 1, 1, 1, 1]
            self.specific2share_lateral_strides = [16, 8, 4, 2, 1]
            self.specific2share_pooling_size = [0, 0, 0, 0, 0]
            if getattr(custom_config, 'PARALLEL_T2B_POOL_SIZE', False):
                self.specific2share_pooling_size = custom_config.PARALLEL_T2B_POOL_SIZE
            if getattr(custom_config, 'PRALLEL_T2B_KERNELS', False):
                self.specific2share_lateral_kernels = custom_config.PRALLEL_T2B_KERNELS
            if getattr(custom_config, 'PRALLEL_T2B_PADDINGS', False):
                self.specific2share_lateral_paddings = custom_config.PRALLEL_T2B_PADDINGS
            if getattr(custom_config, 'PRALLEL_T2B_STRIDES', False):
                self.specific2share_lateral_strides = custom_config.PRALLEL_T2B_STRIDES

            if getattr(custom_config, 'PARALLEL_B2T', False):
                self.share2specific_lateral_kernels = [16 + 2, 8 + 2, 4 + 2, 2 + 2, 1 + 2]
                self.share2specific_lateral_paddings = [1, 1, 1, 1, 1]
                self.share2specific_lateral_strides = [16, 8, 4, 2, 1]
                self.share2specific_last_layer_flag = [False] * 4 + [True]
                self.share2specific_b2t_layer_flag = getattr(custom_config, 'PARALLEL_B2T_LAYER', [True] * 5)

                self.parallel_lateral_adapter = nn.Sequential(
                    *[
                        Lateral_Adapter(self.parallel_output_dims[iii], width,
                                        top2bottom_kernel=self.specific2share_lateral_kernels[iii],
                                        top2bottom_padding=self.specific2share_lateral_paddings[iii],
                                        top2bottom_stride=self.specific2share_lateral_strides[iii],
                                        bottom2top_kernel=self.share2specific_lateral_kernels[iii],
                                        bottom2top_padding=self.share2specific_lateral_paddings[iii],
                                        bottom2top_stride=self.share2specific_lateral_strides[iii],
                                        bottom2top_last_layer=self.share2specific_last_layer_flag[iii],
                                        bottom2top_layer_flag=self.share2specific_b2t_layer_flag[iii],
                                        custom_config=custom_config)
                        for iii in range(len(self.parallel_lateral_layers))
                    ]
                )
            else:
                if getattr(custom_config, 'PARALLEL_T2B_WINDOWATTN', False):
                    self.input_q_dim_list = [width] * len(self.parallel_lateral_layers)
                    self.input_kv_dim_list = [width // 16, width // 8, width // 4, width // 2, width]
                    self.output_qk_dim_list = [width // 16, width // 8, width // 4, width // 2, width]
                    self.output_v_dim_list = [width // 16, width // 8, width // 4, width // 2, width]
                    self.head_qk_dim_list = [48, 48, 64, 64, 64]
                    self.head_v_dim_list = [48, 48, 64, 64, 64]
                    self.window_size_q_list = [1] * len(self.parallel_lateral_layers)
                    self.window_size_kv_list = [16, 8, 4, 2, 3]
                    self.dwconv_kv_list = [True] * len(self.parallel_lateral_layers)
                    self.add_linear_list = [True] * len(self.parallel_lateral_layers)
                    self.output_dim_list = [width] * len(self.parallel_lateral_layers)
                    self.slide_window_list = [False] * 4 + [True] * 1
                    self.slide_window_kernel_list = [0] * 4 + [3] * 1
                    self.slide_window_pad_list = [0] * 4 + [1] * 1
                    self.slide_window_stride_list = [0] * 4 + [1] * 1

                    self.parallel_lateral_adapter = nn.Sequential(
                        *[
                            CrossAttentionLayer_Window(input_q_dim=self.input_q_dim_list[iii],
                                                       input_kv_dim=self.input_kv_dim_list[iii],
                                                       output_qk_dim=self.output_qk_dim_list[iii],
                                                       output_v_dim=self.output_v_dim_list[iii],
                                                       head_qk_dim=self.head_qk_dim_list[iii],
                                                       head_v_dim=self.head_v_dim_list[iii],
                                                       window_size_q=self.window_size_q_list[iii],
                                                       window_size_kv=self.window_size_kv_list[iii],
                                                       dwconv_kv=self.dwconv_kv_list[iii],
                                                       add_linear=self.add_linear_list[iii],
                                                       output_dim=self.output_dim_list[iii],
                                                       slide_window=self.slide_window_list[iii],
                                                       slide_window_kernel=self.slide_window_kernel_list[iii],
                                                       slide_window_pad=self.slide_window_pad_list[iii],
                                                       slide_window_stride=self.slide_window_stride_list[iii],
                                                       custom_config=custom_config
                                                       )
                            for iii in range(len(self.parallel_lateral_layers))
                        ]
                    )
                else:
                    self.parallel_lateral_adapter = nn.Sequential(
                        *[
                            Lateral_Adapter(top_dim=self.parallel_output_dims[iii], bottom_dim=width,
                                            top2bottom_kernel=self.specific2share_lateral_kernels[iii],
                                            top2bottom_padding=self.specific2share_lateral_paddings[iii],
                                            top2bottom_stride=self.specific2share_lateral_strides[iii],
                                            custom_config=custom_config,
                                            top2bottom_pool_kernel=self.specific2share_pooling_size[iii],
                                            )
                            for iii in range(len(self.parallel_lateral_layers))
                        ]
                    )

        self.apply(self._init_weights)

    def build_conv_bn_relu(self, input_dim, output_dim, kernel_size, padding, stride):
        proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            )),
            ('bn', nn.BatchNorm2d(output_dim)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        return proj

    def build_early_conv_block(self, width, custom_config):
        if getattr(custom_config, 'EARLY_CONV_RES', False):
            res_layer_strides = getattr(custom_config, 'EARLY_CONV_RES_STRIDES', [2, 2, 2, 2])
            proj = EarlyconvRes(in_dim=3, output_dim=width, first_conv_k=custom_config.EARLY_CONV_RES_FIRSTCONV_KERNEL,
                                res_block=custom_config.EARLY_CONV_RES_BLOCK,
                                res_layers=custom_config.EARLY_CONV_RES_LAYERS, res_layer_strides=res_layer_strides)
        else:
            proj = nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(
                    3,
                    width // 16,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )),
                ('bn_1', nn.BatchNorm2d(48)),
                ('relu_1', nn.ReLU(inplace=True)),
                ('conv_2', nn.Conv2d(
                    width // 16,
                    width // 8,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )),
                ('bn_2', nn.BatchNorm2d(96)),
                ('relu_2', nn.ReLU(inplace=True)),
                ('conv_3', nn.Conv2d(
                    width // 8,
                    width // 4,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )),
                ('bn_3', nn.BatchNorm2d(192)),
                ('relu_3', nn.ReLU(inplace=True)),
                ('conv_4', nn.Conv2d(
                    width // 4,
                    width // 2,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )),
                ('bn_4', nn.BatchNorm2d(384)),
                ('relu_4', nn.ReLU(inplace=True)),
                ('conv_5', nn.Conv2d(
                    width // 2,
                    width,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )),
                ('bn_5', nn.BatchNorm2d(width)),
                ('relu_5', nn.ReLU(inplace=True)),
                ('conv_6', nn.Conv2d(
                    width,
                    width,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False,
                )),
            ]))
        return proj

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, action=None, h=None, w=None, class_embed=None, position_embed=None, ln_pre=None,
                original_input=None, output_layer_fea=False, output_last_attnmap=False):
        if output_layer_fea:
            output_feats_list = []
        if output_last_attnmap:
            output_attnmap_list = []
        if action is not None:
            assert not output_layer_fea, 'not implemented'
            assert self.gumbel_select
            assert len(self.resblocks) == len(action)
            for idx, resblock in enumerate(self.resblocks):
                if self.first_conv and idx == 0:
                    x = resblock(x)  # shape = [*, width, grid, grid]
                    b, c, h, w = x.size()
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                    x = torch.cat([class_embed.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                         device=x.device), x],
                                  dim=1)  # shape = [*, grid ** 2 + 1, width]
                    x = x + position_embed.to(x.dtype)
                    x = ln_pre(x)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                else:
                    if output_last_attnmap and idx == len(self.resblocks) - 1:
                        x, last_attnmap = resblock(x, action=action[idx, :], h=h, w=w, output_last_attnmap=True)
                    else:
                        x = resblock(x, action=action[idx, :], h=h, w=w)
                if output_layer_fea:
                    output_feats_list.append(x.data.cpu())
            # return x
        else:
            for idx, resblock in enumerate(self.resblocks):
                if (self.perceiver_v_flag or self.perceiver_t_flag) and idx in self.perceiver_layer:
                    perceiver_layer_idx = self.perceiver_layer.index(idx)
                    if perceiver_layer_idx == 0:
                        perceiver_latent_input = self.perceiver_latent.unsqueeze(1).repeat(1, x.size(1), 1)
                    x = self.perceiver_latent2ori_blocks[perceiver_layer_idx](x, perceiver_latent_input,
                                                                              perceiver_latent_input, h=h, w=w)
                    # if output_last_attnmap and idx == len(self.resblocks) - 1:
                    if output_last_attnmap:
                        x, last_attnmap = resblock(x, h=h, w=w, output_last_attnmap=True)
                    else:
                        x = resblock(x, h=h, w=w)
                elif self.first_conv and idx == 0:
                    # assert idx not in self.parallel_lateral_layers
                    if self.parallel_reuse_earlyconv_firstlayer:
                        early_conv_output_1 = resblock[:3](x)
                        x = resblock[3:](early_conv_output_1)
                    elif self.parallel_reuse_earlyconv_alllayer:
                        early_conv_output_1 = resblock[:3](x)
                        early_conv_output_2 = resblock[3:6](early_conv_output_1)
                        early_conv_output_3 = resblock[6:9](early_conv_output_2)
                        early_conv_output_4 = resblock[9:12](early_conv_output_3)
                        early_conv_output_5 = resblock[12:15](early_conv_output_4)
                        x = resblock[15:](early_conv_output_5)
                        early_conv_output_dict = {0: early_conv_output_1, 1: early_conv_output_2,
                                                  2: early_conv_output_3, \
                                                  3: early_conv_output_4, 4: early_conv_output_5}
                    else:
                        x = resblock(x)  # shape = [*, width, grid, grid]
                        # pdb.set_trace()
                    b, c, h, w = x.size()
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                    x = torch.cat([class_embed.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                         device=x.device), x],
                                  dim=1)  # shape = [*, grid ** 2 + 1, width]
                    x = x + position_embed.to(x.dtype)
                    x = ln_pre(x)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    if output_layer_fea:
                        layer_fea_output = x
                elif self.parallel_in_v and idx in self.parallel_lateral_layers:
                    lateral_idx = self.parallel_lateral_layers.index(idx)
                    if lateral_idx == 0:
                        if self.parallel_reuse_earlyconv_firstlayer:
                            original_input = early_conv_output_1
                        elif self.parallel_reuse_earlyconv_alllayer:
                            original_input = early_conv_output_dict[0]
                        parallel_x = self.parallel_branch_v[lateral_idx](original_input)
                    else:
                        if self.parallel_reuse_earlyconv_alllayer:
                            parallel_x = self.parallel_branch_v[lateral_idx](parallel_x) + early_conv_output_dict[
                                lateral_idx]
                        else:
                            parallel_x = self.parallel_branch_v[lateral_idx](parallel_x)
                    parallel_x, x = self.parallel_lateral_adapter[lateral_idx](top_input=parallel_x, bottom_input=x,
                                                                               h=h, w=w)
                    # if output_last_attnmap and idx == len(self.resblocks) - 1:
                    if output_last_attnmap:
                        x, last_attnmap = resblock(x, h=h, w=w, output_last_attnmap=True)
                    elif output_layer_fea:
                        x, layer_fea_output = resblock(x, h=h, w=w, output_layer_fea=True)
                    else:
                        x = resblock(x, h=h, w=w)
                else:
                    # if output_last_attnmap and idx == len(self.resblocks) - 1:
                    if output_last_attnmap:
                        x, last_attnmap = resblock(x, h=h, w=w, output_last_attnmap=True)
                    elif output_layer_fea:
                        x, layer_fea_output = resblock(x, h=h, w=w, output_layer_fea=True)
                    else:
                        x = resblock(x, h=h, w=w)
                if output_layer_fea:
                    output_feats_list.append(layer_fea_output.data.cpu())
                if output_last_attnmap and 'last_attnmap' in locals():
                    output_attnmap_list.append(last_attnmap.data.cpu())
            # return x
        if output_layer_fea:
            return x, output_feats_list
        if output_last_attnmap:
            # return x, last_attnmap
            return x, output_attnmap_list

        return x
        # Before 0719: should be no difference
        # return self.resblocks(x, h=h, w=w)


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 pool_type: str = 'default',
                 skip_cls: bool = False,
                 drop_path: float = 0.0,
                 gumbel_select=False,
                 gumbel_addtwo=False,
                 custom_config=None, ):
        super().__init__()
        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.early_conv = getattr(custom_config, 'EARLY_CONV', False)
        self.early_conv_new_implement = getattr(custom_config, 'EARLY_CONV_NEW_IMPLEMENT', False)
        self.parallel_in_v = getattr(custom_config, 'PARALLEL_IN_V', False)
        if self.early_conv:
            if not self.early_conv_new_implement:
                self.conv1 = self.build_early_conv_block(width)
        else:
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False
            )
        if getattr(custom_config, 'VISUAL_LAYER_MINUS1', False):
            assert self.early_conv
            layers = layers - 1

        self.sequence_length = (input_resolution // patch_size) ** 2 + 1

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            if (not self.skip_cls):
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length, stride=self.sequence_length,
                                           groups=width)
            else:
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length - 1, stride=self.sequence_length,
                                           groups=width)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                self.sequence_length, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, gumbel_select=gumbel_select, gumbel_addtwo=gumbel_addtwo,
            custom_config=custom_config, modality='visual', first_conv=self.early_conv_new_implement
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # >>>>>>>>>>>>> For NMI OUTPUT LAST LN <<<<<<<<<<<<<<<<
        self.output_last_ln = getattr(custom_config, 'OUTPUT_LAST_LN', False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_early_conv_block(self, width):
        proj = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(
                3,
                width // 16,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )),
            ('bn_1', nn.BatchNorm2d(48)),
            ('relu_1', nn.ReLU(inplace=True)),
            ('conv_2', nn.Conv2d(
                width // 16,
                width // 8,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )),
            ('bn_2', nn.BatchNorm2d(96)),
            ('relu_2', nn.ReLU(inplace=True)),
            ('conv_3', nn.Conv2d(
                width // 8,
                width // 4,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )),
            ('bn_3', nn.BatchNorm2d(192)),
            ('relu_3', nn.ReLU(inplace=True)),
            ('conv_4', nn.Conv2d(
                width // 4,
                width // 2,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )),
            ('bn_4', nn.BatchNorm2d(384)),
            ('relu_4', nn.ReLU(inplace=True)),
            ('conv_5', nn.Conv2d(
                width // 2,
                width,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )),
            ('bn_5', nn.BatchNorm2d(width)),
            ('relu_5', nn.ReLU(inplace=True)),
            ('conv_6', nn.Conv2d(
                width,
                width,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
            )),
        ]))
        return proj

    def forward(self, x: torch.Tensor, action=None, output_layer_fea=False, output_last_attnmap=False):
        if self.early_conv and self.early_conv_new_implement:
            if self.parallel_in_v:
                original_x = x.clone()
            else:
                original_x = None
            if output_layer_fea:
                x, layer_feas = self.transformer(x, action=action, class_embed=self.class_embedding,
                                                 position_embed=self.positional_embedding, ln_pre=self.ln_pre,
                                                 output_layer_fea=True, original_input=original_x)
            elif output_last_attnmap:
                x, last_attnmap = self.transformer(x, action=action, class_embed=self.class_embedding,
                                                   position_embed=self.positional_embedding, ln_pre=self.ln_pre,
                                                   output_last_attnmap=True, original_input=original_x)
            else:
                x = self.transformer(x, action=action, class_embed=self.class_embedding,
                                     position_embed=self.positional_embedding, ln_pre=self.ln_pre,
                                     original_input=original_x)
        else:
            if self.parallel_in_v:
                original_x = x.clone()
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            B, C, H, W = x.size()
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device), x],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            if self.parallel_in_v:
                if output_layer_fea:
                    x, layer_feas = self.transformer(x, action=action, h=H, w=W, original_input=original_x,
                                                     output_layer_fea=True)
                elif output_last_attnmap:
                    x, last_attnmap = self.transformer(x, action=action, h=H, w=W, original_input=original_x,
                                                       output_last_attnmap=True)
                else:
                    x = self.transformer(x, action=action, h=H, w=W, original_input=original_x)
            else:
                if output_layer_fea:
                    x, layer_feas = self.transformer(x, action=action, h=H, w=W, output_layer_fea=True)
                elif output_last_attnmap:
                    x, last_attnmap = self.transformer(x, action=action, h=H, w=W, output_last_attnmap=True)
                else:
                    x = self.transformer(x, action=action, h=H, w=W)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.output_last_ln:
            layer_feas.append(self.ln_post(x.permute(1, 0, 2)).data.cpu())

        if (self.pool_type == 'average'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = torch.mean(x, dim=1)
        elif (self.pool_type == 'linear'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[:, 0, :]

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        if output_layer_fea:
            return x, layer_feas
        if output_last_attnmap:
            return x, last_attnmap

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_drop_path: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_style: str = 'clip',
                 gather_tensors: bool = False,
                 eot_token: int = None,
                 tokenizer_style: str = 'clip',
                 pool_type: str = 'default',
                 skip_cls: bool = False,
                 custom_config=None,
                 output_dir=None,
                 ):
        super().__init__()

        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.tokenizer_style = tokenizer_style
        self.eot_token = eot_token
        self.context_length = context_length
        self.transformer_style = transformer_style
        self.transformer_width = transformer_width
        self.vocab_size = vocab_size
        self.gather_tensors = gather_tensors
        self.custom_config = custom_config
        self.gumbel_select = getattr(custom_config, 'GUMBEL_SELECT', False)
        self.gumbel_addtwo = getattr(custom_config, 'GUMBEL_ADDTWO', False)
        self.share_bottom_layer = getattr(custom_config, 'SHARE_BOTTOM_LAYER', False)
        self.save_grad = getattr(custom_config, 'SAVE_GRADIENT', False) or getattr(custom_config,
                                                                                   'GET_GRADIENT_FROMCKPT', False)

        if self.gumbel_select:
            assert vision_layers == transformer_layers
            # self.gumbel_logit = nn.Parameter(torch.randn(vision_layers, 2))
            self.gumbel_logit = nn.Parameter(1e-3 * torch.randn(vision_layers, 2))
            self.tau = 5

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                pool_type=self.pool_type,
                skip_cls=self.skip_cls,
                drop_path=vision_drop_path,
                gumbel_select=self.gumbel_select,
                gumbel_addtwo=self.gumbel_addtwo,
                custom_config=custom_config,
            )

        if self.transformer_style == 'clip':
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                gumbel_select=self.gumbel_select,
                gumbel_addtwo=self.gumbel_addtwo,
                custom_config=custom_config,
                modality='text',
            )

            shared_modules = getattr(custom_config, 'SHARE_MODULES', None)
            if shared_modules is not None:
                load_arch_file = getattr(custom_config, 'LOAD_SEARCHED_ARCH', None)
                if load_arch_file is not None:
                    load_arch_path = Path(output_dir)
                    load_arch_path = load_arch_path / load_arch_file
                    load_arch_path = str(load_arch_path)
                    arch_dict = torch.load(load_arch_path)
                    arch_logits = arch_dict['saved_arch']['best']
                    logging.info('Share modules: {}'.format(shared_modules))
                    logging.info('loaded arch path: {}'.format(load_arch_path))
                    logging.info('loaded arch logits: {}'.format(arch_logits))
                    for m in shared_modules:
                        for i, block in enumerate(self.visual.transformer.resblocks):
                            if arch_logits[i, 0] > arch_logits[i, 1]:
                                setattr(self.transformer.resblocks[i], m, getattr(block, m))
                else:
                    shared_n_layers = getattr(custom_config, 'N_LAYERS', -1)
                    logging.info('Share modules: {}'.format(shared_modules))
                    logging.info('Shared n layers: {}'.format(shared_n_layers))
                    # import pdb
                    # pdb.set_trace()
                    for m in shared_modules:
                        for i, block in enumerate(self.visual.transformer.resblocks):
                            assert len(self.visual.transformer.resblocks) >= shared_n_layers
                            if self.share_bottom_layer:
                                if shared_n_layers != -1 and i >= shared_n_layers:
                                    continue
                            else:
                                if shared_n_layers != -1 and i < shared_n_layers:
                                    continue
                            if len(m.split('.')) != 1:
                                m_groups = m.split('.')
                                if m_groups[0] == 'attn':
                                    if getattr(custom_config, 'VISUAL_LAYER_MINUS1', False):
                                        setattr(self.transformer.resblocks[i + 1].attn, m_groups[1],
                                                getattr(block.attn, m_groups[1]))
                                    else:
                                        setattr(self.transformer.resblocks[i].attn, m_groups[1],
                                                getattr(block.attn, m_groups[1]))
                            else:
                                if getattr(custom_config, 'VISUAL_LAYER_MINUS1', False):
                                    setattr(self.transformer.resblocks[i + 1], m, getattr(block, m))
                                else:
                                    setattr(self.transformer.resblocks[i], m, getattr(block, m))
                # import pdb
                # pdb.set_trace()

            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, transformer_width)
            )
            trunc_normal_(self.positional_embedding, std=.02)

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            self.conv_pool = nn.Conv1d(self.transformer_width, self.transformer_width, self.context_length,
                                       stride=self.context_length, groups=self.transformer_width)

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        self.apply(self._init_weights)
        if getattr(custom_config, 'LORA_INIT', False):
            self.init_lora(getattr(custom_config, 'LORA_INIT', False))

        if getattr(custom_config, 'CONVIT_IN_V', False):
            self.init_convit(vision_layers, logging)

    def init_lora(self, init_method):
        assert init_method == 'v1'
        for n, p in self.named_parameters():
            if 'proj_adapter1.weight' in n:
                nn.init.normal_(p, std=0.02)
            if 'proj_adapter2.weight' in n:
                p.data.zero_()

    def init_convit(self, n_layers, logging):
        self.convit_layers = getattr(self.custom_config, 'CONVIT_LAYERS', [])
        if len(self.convit_layers) == 0:
            self.convit_layers = list(range(n_layers))
        for ii in range(n_layers):
            if ii in self.convit_layers:
                self.visual.transformer.resblocks[ii].attn.convit_local_init()
        logging.info('Finished initilization of convit_local_init, {} layers in total'.format(n_layers))

    def get_arch_param(self):
        return [self.gumbel_logit]

    def get_arch_data(self):
        return self.gumbel_logit.data.cpu()

    def set_tau(self, tau):
        self.tau = tau

    def show_arch(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                torch.nn.functional.softmax(self.gumbel_logit, dim=-1).cpu()
            )

    def gumbel_softmax(self, logits):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        while True:
            gumbels = -torch.empty_like(logits).exponential_().log()
            total_logits = (logits + gumbels) / self.tau
            # Before 0706:
            # total_logits = (logits.log_softmax(dim=1) + gumbels) / self.tau
            probs = nn.functional.softmax(total_logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(total_logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
            ):
                continue
            else:
                break
        return hardwts

    def archi_softmax(self, logits):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        while True:
            probs = nn.functional.softmax(logits / self.tau, dim=1)
            # total_logits = (logits.log_softmax(dim=1)) / self.tau
            # probs = nn.functional.softmax(total_logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                    (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
            ):
                continue
            else:
                break
        return hardwts

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
            'logit_scale'
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if getattr(self.custom_config, 'T2B_WINDOWATTN_RELATIVE_POS_NOWD', False):
            return {'relative_position_bias_table'}
        else:
            return {}

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.positional_embedding.dtype
        # After 07/20, conv1 might be a NNlist
        # return self.visual.conv1.weight.dtype

    def encode_image(self, image, norm=True, action=None):
        x = self.visual(image.type(self.dtype), action=action)

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def output_image_attnmap(self, image, norm=True, action=None):
        x, last_attnmap = self.visual(image.type(self.dtype), action=action, output_last_attnmap=True)
        return last_attnmap

    def output_text_attnmap(self, text, norm=True, action=None):
        # 'text' is not the raw text, it is the tokens.

        assert self.transformer_style == 'clip'
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, last_attnmap = self.transformer(x, action=action, output_last_attnmap=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return last_attnmap

    def output_image_inter_feature(self, image, norm=True, action=None):
        x, feats_layer_list = self.visual(image.type(self.dtype), action=action, output_layer_fea=True)

        return feats_layer_list

    def output_text_inter_feature(self, text, norm=True, action=None):
        # 'text' is not the raw text, it is the tokens.

        assert self.transformer_style == 'clip'
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, feats_layer_list = self.transformer(x, action=action, output_layer_fea=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if getattr(self.custom_config, 'OUTPUT_LAST_LN', False):
            feats_layer_list.append(self.ln_final(x.permute(1, 0, 2)).data.cpu())

        if (self.pool_type == 'default'):
            if self.tokenizer_style == 'clip':
                x = x[
                    torch.arange(x.shape[0]),
                    text.argmax(dim=-1)
                ]
        elif (self.pool_type == 'linear'):
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[
                torch.arange(x.shape[0]),
                :
                ]
            x = torch.mean(x, dim=1)
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        if norm:
            x = x / x.norm(dim=-1, keepdim=True)
        return feats_layer_list

    def encode_text(self, text, norm=True, action=None):
        # 'text' is not the raw text, it is the tokens.

        if self.transformer_style == 'clip':
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, action=action)
            x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if (self.pool_type == 'default'):
            if self.tokenizer_style == 'clip':
                x = x[
                    torch.arange(x.shape[0]),
                    text.argmax(dim=-1)
                ]
        elif (self.pool_type == 'linear'):
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[
                torch.arange(x.shape[0]),
                :
                ]
            x = torch.mean(x, dim=1)

        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    # def encode_image_with_action(self, image, norm=True):
    #     assert self.gumbel_select
    #     action = gumbel_softmax(self.gumbel_logit)
    #     features_image = self.encode_image(image, action=action)
    #     return features_image

    # def encode_text_with_action(self, text, norm=True):
    #     assert self.gumbel_select
    #     action = gumbel_softmax(self.gumbel_logit)
    #     features_text = self.encode_text(text, action=action)
    #     return features_text

    def generate_action(self, gumbel_sample=True):
        if gumbel_sample:
            action = self.gumbel_softmax(self.gumbel_logit)
        else:
            action = self.archi_softmax(self.gumbel_logit)
        return action

    # def forward_savegrad(self, image, text):
    #     if self.gumbel_select:
    #         action = self.gumbel_softmax(self.gumbel_logit)
    #     else:
    #         action = None

    #     features_image = self.encode_image(image, action=action)
    #     features_text = self.encode_text(text, action=action)

    #     # cosine similarity as logits
    #     T = self.logit_scale.exp()

    #     if self.gather_tensors:
    #         features_image_all = gather_tensors(features_image)
    #         features_text_all = gather_tensors(features_text)
    #         logits_image_text = T * features_image_all @ features_text_all.t()
    #         logits_image_text_fiximage = T * features_image_all.detach() @ features_text_all.t()
    #         logits_image_text_fixtext = T * features_image_all @ features_text_all.t().detach()
    #     else:
    #         logits_image_text = T * features_image @ features_text.t()
    #         logits_image_text_fiximage = T * features_image.detach() @ features_text.t()
    #         logits_image_text_fixtext = T * features_image @ features_text.t().detach()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_image_text, logits_image_text_fiximage, logits_image_text_fixtext

    def forward(self, image, text):
        if self.gumbel_select:
            action = self.gumbel_softmax(self.gumbel_logit)
        else:
            action = None

        features_image = self.encode_image(image, action=action)
        features_text = self.encode_text(text, action=action)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        if self.gather_tensors:
            features_image_all = gather_tensors(features_image)
            features_text_all = gather_tensors(features_text)
            logits_image_text = T * features_image_all @ features_text_all.t()
            if self.save_grad:
                logits_image_text_fiximage = T.detach() * features_image_all.detach() @ features_text_all.t()
                logits_image_text_fixtext = T * features_image_all @ features_text_all.t().detach()
        else:
            logits_image_text = T * features_image @ features_text.t()
            if self.save_grad:
                logits_image_text_fiximage = T.detach() * features_image.detach() @ features_text.t()
                logits_image_text_fixtext = T * features_image @ features_text.t().detach()

                # shape = [global_batch_size, global_batch_size]
        if self.save_grad:
            return logits_image_text, logits_image_text_fiximage, logits_image_text_fixtext
        else:
            return logits_image_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def get_clip_model(config, vocab_size=None, eot_token=None, **kwargs):
    embed_dim = config.MODEL.SPEC.EMBED_DIM

    image_resolution = config.TRAIN.IMAGE_SIZE[0]
    spec_vision = config.MODEL.SPEC.VISION
    vision_width = spec_vision.WIDTH
    vision_drop_path = getattr(spec_vision, 'DROP_PATH', 0.0)

    if (spec_vision.MODEL == 'vit'):
        vision_layers = spec_vision.LAYERS
        vision_patch_size = spec_vision.PATCH_SIZE
    else:
        vision_layers = tuple(spec_vision.LAYERS)
        vision_patch_size = None

    spec_text = config.MODEL.SPEC.TEXT
    context_length = spec_text.CONTEXT_LENGTH

    # Allow information from the tokenizer class,
    # which is contained in the dataloader, to pass
    # into our model definition
    if (vocab_size is None):
        vocab_size = spec_text.VOCAB_SIZE
    else:
        vocab_size = vocab_size

    transformer_width = spec_text.WIDTH
    transformer_heads = spec_text.HEADS
    transformer_layers = spec_text.LAYERS
    transformer_style = spec_text.STYLE
    tokenizer_style = spec_text.TOKENIZER

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        vision_drop_path, context_length, vocab_size, transformer_width,
        transformer_heads, transformer_layers, transformer_style,
        gather_tensors=getattr(config.MODEL.SPEC, 'GATHER_TENSORS', False),
        eot_token=eot_token, tokenizer_style=tokenizer_style,
        pool_type=getattr(config.MODEL.SPEC, 'POOL_TYPE', 'default'),
        skip_cls=getattr(config.MODEL.SPEC, 'SKIP_CLS', False),
        custom_config=config.CUSTOM,
        output_dir=config.OUTPUT_DIR
    )

    return model
