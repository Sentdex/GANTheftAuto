"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
SPADE is modified from https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
Licensed under the CC BY-NC-SA 4.0 license:
https://github.com/NVlabs/SPADE/blob/master/LICENSE.md
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as SN

import sys
import utils
sys.path.append('..')



class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def choose_netG_encoder(basechannel=512, opts=None):
    '''
    image input encoder
    '''
    if opts.img_size[0] == 64:
        last_dim = basechannel
        enc = nn.Sequential(
            nn.Conv2d(opts.num_channel, basechannel // 8, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 8, basechannel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 8, basechannel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 8, basechannel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            View((-1, (basechannel // 8) * 7 * 7)),
            nn.Linear((basechannel // 8) * 7 * 7, last_dim),
            nn.LeakyReLU(0.2),
        )
    elif opts.img_size[0] == 84:
        chan = opts.encoder_chan_multiplier * basechannel
        enc = nn.Sequential(
            nn.Conv2d(opts.num_channel, chan // 8, 3, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan // 8, chan // 8, 3, 1, (0, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan // 8, chan // 8, 3, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan // 8, chan // 8, 3, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan // 8, chan // 8, 3, 2, 0),
            nn.LeakyReLU(0.2),
            View((-1, (chan // 8) * 8 * 8)),
            nn.Linear((chan // 8) * 8 * 8, basechannel),
            nn.LeakyReLU(0.2),
        )
    elif opts.img_size[0] == 128:
        last_dim = basechannel
        enc = nn.Sequential(
            nn.Conv2d(opts.num_channel, basechannel // 16, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 16, basechannel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 8, basechannel // 4, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 4, basechannel // 2, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basechannel // 2, basechannel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            View((-1, (basechannel // 8) * 7 * 7)),
            nn.Linear((basechannel // 8) * 7 * 7, last_dim),
            nn.LeakyReLU(0.2),
        )
    else:
        assert 0, 'model-%s not supported'

    return enc


def choose_netD_temporal(opts, conv3d_dim, window=[]):
    '''
    temporal discriminator
    '''
    in_dim = opts.nfilterD * 16
    extractors, finals = [], []

    # temporarily hand-designed for steps 6 / 12 / 18
    first_spatial_filter = 2
    if opts.img_size[0] == 84:
        first_spatial_filter = 3
    if utils.check_arg(opts, 'simple_blocks'):
        net1 = nn.Sequential(
            SN(nn.Conv3d(in_dim, conv3d_dim, (2, 2, 2), (1, 1, 1))),
            nn.LeakyReLU(0.2),
            SN(nn.Conv3d(conv3d_dim, conv3d_dim * 2, (3, 2, 2), (2, 1, 1))),
            nn.LeakyReLU(0.2)
        )
        head1 = nn.Sequential(
            SN(nn.Conv3d(conv3d_dim * 2, 1, (2, 1, 1), (2, 1, 1))),
        )
        extractors.append(net1)
        finals.append(head1)

        if window > 6:  # 18
            net2 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 2, conv3d_dim * 4, (3, 1, 1), (2, 1, 1))),
                nn.LeakyReLU(0.2),
            )
            head2 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 4, 1, (2, 1, 1))),
            )
            extractors.append(net2)
            finals.append(head2)

        if window > 18:  # 32
            net3 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 4, conv3d_dim * 8, (3, 1, 1), (2, 1, 1))),
                nn.LeakyReLU(0.2),
            )
            head3 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 8, 1, (3, 1, 1))),
            )
            extractors.append(net3)
            finals.append(head3)
    elif 'sn' in opts.D_temp_mode:
        net1 = nn.Sequential(
            SN(nn.Conv3d(in_dim, conv3d_dim, (2, first_spatial_filter, first_spatial_filter), (1, 1, 1))),
            nn.LeakyReLU(0.2),
            SN(nn.Conv3d(conv3d_dim, conv3d_dim * 2, (3, 3, 3), (2, 1, 1))),
            nn.LeakyReLU(0.2)
        )
        head1 = nn.Sequential(
            SN(nn.Conv3d(conv3d_dim * 2, 1, (2, 1, 1), (1, 1, 1))),
        )
        extractors.append(net1)
        finals.append(head1)

        if window >= 12:  # 12
            net2 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 2, conv3d_dim * 4, (3, 1, 1), (1, 1, 1))),
                nn.LeakyReLU(0.2),
            )
            head2 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 4, 1, (3, 1, 1))),
            )
            extractors.append(net2)
            finals.append(head2)

        if window >= 18:  # 18
            net3 = nn.Sequential(
                SN(nn.Conv3d(conv3d_dim * 4, conv3d_dim * 8, (3, 1, 1), (2, 1, 1))),
                nn.LeakyReLU(0.2),
            )
            if window == 18 or window == 28:
                head3 = nn.Sequential(
                    SN(nn.Conv3d(conv3d_dim * 8, 1, (2, 1, 1), (1, 1, 1))),
                )
            else:
                head3 = nn.Sequential(
                    SN(nn.Conv3d(conv3d_dim * 8, 1, (4, 1, 1), (2, 1, 1))),
                )
            extractors.append(net3)
            finals.append(head3)
    else:
        net1 = nn.Sequential(
            nn.Conv3d(in_dim, conv3d_dim, (2, 3, 3), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(conv3d_dim, conv3d_dim * 2, (3, 3, 3), (2, 1, 1)),
            nn.LeakyReLU(0.2)
        )
        head1 = nn.Sequential(
            nn.Conv3d(conv3d_dim * 2, 1, (2, 1, 1), (1, 1, 1)),
        )
        extractors.append(net1)
        finals.append(head1)

        if window >= 12:  # 12
            net2 = nn.Sequential(
                nn.Conv3d(conv3d_dim * 2, conv3d_dim * 4, (3, 1, 1), (1, 1, 1)),
                nn.LeakyReLU(0.2),
            )
            head2 = nn.Sequential(
                nn.Conv3d(conv3d_dim * 4, 1, (3, 1, 1)),
            )
            extractors.append(net2)
            finals.append(head2)

        if window >= 18:  # 18
            net3 = nn.Sequential(
                nn.Conv3d(conv3d_dim * 4, conv3d_dim * 8, (2, 1, 1), (2, 1, 1)),
                nn.LeakyReLU(0.2),
            )
            head3 = nn.Sequential(
                nn.Conv3d(conv3d_dim * 8, 1, (3, 1, 1)),
            )
            extractors.append(net3)
            finals.append(head3)

    return extractors, finals


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, ks=3, nhidden=128):
        super().__init__()
        norm_nc = int(norm_nc)
        label_nc = int(label_nc)
        ks = ks

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, segmap):
        if segmap is None:
            return self.param_free_norm(x)
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)

        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return self.activation(out)
