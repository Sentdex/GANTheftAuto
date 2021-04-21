"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
Contains some code from:
https://github.com/ajbrock/BigGAN-PyTorch
with the following license:

MIT License

Copyright (c) 2019 Andy Brock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import utils
import sys
sys.path.append('..')
import torch.nn.utils.spectral_norm as SN

from simulator_model.model_utils import View
from simulator_model import model_utils
from simulator_model import layers
import functools
from torch.nn import init



class DiscriminatorSingle(nn.Module):
    '''
    BigGAN discriminator architecture from
    https://github.com/ajbrock/BigGAN-PyTorch
    '''
    def __init__(self, D_ch=64, D_wide=False, resolution=(64, 64),
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', opts=None,  **kwargs):
        super(DiscriminatorSingle, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = '64_32' if (resolution[0] == 96 and resolution[1] == 160) or (resolution[0] == 48 and resolution[1] == 80) else D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = self.D_arch(self.ch, self.attention)[f'{resolution[1]}x{resolution[0]}']
        self.opts = opts
        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)


        # Initialize weights
        if not skip_init:
            self.init_weights()

    def D_arch(self, ch=64, attention='64', ksize='333333', dilation='111111'):
        arch = {}
        arch['160x96'] = {'in_channels': [3] + [ch * item for item in [1, 1, 2, 2, 4, 8, 16]],
                          'out_channels': [item * ch for item in [1, 1, 2, 2, 4, 8, 16, 32]],
                          'downsample': [False, True, False] * 1 + [True] * 4 + [False] * 1,
                          'resolution': [64, 32, 16, 8, 4, 4, 4, 4, 4],
                          'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                        for i in range(2, 8)}}
        arch['80x48'] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 4, 4, 8, 8]],
                          'out_channels': [item * ch for item in [1, 2, 4, 4, 4, 8, 8, 16]],
                          'downsample': [True] * 2 + [False] * 2 + [True] * 2 + [False] * 1 + [False] * 1,
                          'resolution': [64, 32, 16, 8, 4, 4, 4, 4],
                          'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                        for i in range(2, 8)}}
        arch['128x128'] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8]],
                           'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
                           'downsample': [True] * 5 + [False],
                           'resolution': [64, 32, 16, 8, 4, 4],
                           'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                         for i in range(2, 8)}}
        arch['84x84'] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
                         'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                         'downsample': [True] * 4 + [False],
                         'resolution': [32, 16, 8, 4, 4],
                         'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                       for i in range(2, 7)}}
        arch['64x64'] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
                         'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                         'downsample': [True] * 4 + [False],
                         'resolution': [32, 16, 8, 4, 4],
                         'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                       for i in range(2, 7)}}
        return arch

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = self.activation(h)
        # Apply global sum pooling as in SN-GAN
        single_d = torch.sum(h, [2, 3])
        # Get initial class-unconditional output
        out = self.linear(single_d)

        return out, h




class Discriminator(nn.Module):
    '''
    GameGAN discriminator: single, action-conditioned, temporal
    '''
    def __init__(self, opts, nfilter=32, nfilter_max=1024):
        super(Discriminator, self).__init__()

        self.opts = opts
        self.simple_blocks = utils.check_arg(self.opts, 'simple_blocks')

        # single frame discrimiantor
        f_size = 4
        if self.opts.img_size[0] == 84:
            f_size = 5
        elif (self.opts.img_size[0] == 96 and self.opts.img_size[1] == 160) or (self.opts.img_size[0] == 48 and self.opts.img_size[1] == 80):
            f_size = (3, 5)
        if self.simple_blocks:
            f_size = 3
            expand_dim=opts.nfilterD * 4
            if opts.img_size[0] == 128:
                self.ds = nn.Sequential(
                    SN(nn.Conv2d(3, expand_dim, 4, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim, expand_dim * 2, 3, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim * 2, expand_dim * 4, 3, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim * 4, expand_dim * 4, 3, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim * 4, expand_dim * 4, 3, 2)),
                    nn.LeakyReLU(0.2),
                    View((-1, expand_dim * 4, 3, 3))
                )
            else:
                self.ds = nn.Sequential(
                    SN(nn.Conv2d(3, expand_dim, 4, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim, expand_dim * 2, 3, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim * 2, expand_dim * 4, 3, 2)),
                    nn.LeakyReLU(0.2),
                    SN(nn.Conv2d(expand_dim * 4, expand_dim * 4, 3, 2)),
                    nn.LeakyReLU(0.2),
                    View((-1, expand_dim * 4, 3, 3))
                )
            # patch level logits
            self.single_frame_discriminator_patch = nn.Sequential(
                SN(nn.Conv2d(expand_dim * 4, expand_dim * 4, 2, 1, 1)),
                nn.LeakyReLU(0.2),
                SN(nn.Conv2d(expand_dim * 4, 1, 2, 1)),
                View((-1, 1, 3, 3)))
            # single logit for entire image
            self.single_frame_discriminator_all = nn.Sequential(
                SN(nn.Conv2d(expand_dim * 4, expand_dim * 4, 2, 1)),
                nn.LeakyReLU(0.2),
                SN(nn.Conv2d(expand_dim * 4, 1, 2, 1)),
                View((-1, 1)))
            conv3d_dim = self.opts.nfilterD_temp
        else:
            # bigGAN discriminator architecture
            self.ds = DiscriminatorSingle(D_ch=opts.nfilterD, opts=opts, resolution=self.opts.img_size)
            conv3d_dim = self.opts.nfilterD_temp

        # temporal discriminator
        self.temporal_window = self.opts.config_temporal
        self.conv3d, self.conv3d_final = \
            model_utils.choose_netD_temporal(
                self.opts, conv3d_dim, window=self.temporal_window
            )
        self.conv3d = nn.ModuleList(self.conv3d)
        self.conv3d_final = nn.ModuleList(self.conv3d_final)

        self.which_conv = functools.partial(layers.SNConv2d,
                                            kernel_size=f_size, padding=0,
                                            num_svs=1, num_itrs=1,
                                            eps=1e-12)
        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=1e-12)

        # action-conditioned discriminator
        self.trans_conv = self.which_conv(opts.nfilterD*16*2 * (2 if opts.img_size[0] == 96 and opts.img_size[1] == 160 else 1), 256)
        self.action_linear1 = self.which_linear(512, 512)
        self.action_linear_out = self.which_linear(512, 1)

        action_space = 10 if not utils.check_arg(self.opts, 'action_space') else self.opts.action_space
        self.action_to_feat = nn.Linear(action_space, 256)
        self.to_transition_feature = nn.Sequential(self.trans_conv,
                                                   nn.LeakyReLU(0.2),
                                                   View((-1, 256)))
        self.action_discriminator = nn.Sequential(self.action_linear1,
                                                    nn.LeakyReLU(0.2),
                                                    self.action_linear_out)
        self.reconstruct_action_z = nn.Sequential(nn.Linear(256, action_space + self.opts.z), )  # 4, 1, 0),


    def forward(self, images, actions, states, warm_up, sample=None, neg_actions=None, rev_steps=0,
                epoch=0, step=0):
        neg_action_predictions, rev_predictions, content_predictions = None, None, []
        batch_size = actions[0].size(0)
        if warm_up == 0:
            warm_up = 1 # even if warm_up is 0, the first screen is from GT

        # run single frame discriminator
        gt_states = torch.cat(states[:warm_up], dim=0)
        single_frame_predictions_patch = None
        if self.simple_blocks:
            tmp_gt = self.ds(gt_states)
            tmp_gen = self.ds(images)
            tmp_features = torch.cat([tmp_gt, tmp_gen], dim=0)
            single_frame_predictions_patch = self.single_frame_discriminator_patch(tmp_gen)
            single_frame_predictions_all = self.single_frame_discriminator_all(tmp_gen)
        else:
            single_frame_predictions_all, tmp_features = self.ds(torch.cat([gt_states, images], dim=0))
            single_frame_predictions_all = single_frame_predictions_all[warm_up*batch_size:]

        frame_features = tmp_features[warm_up*batch_size:]

        # run action-conditioned discriminator and reconstruct action, z
        prev_frames = torch.cat([tmp_features[:warm_up*batch_size],
                                 tmp_features[(warm_up+warm_up-1)*batch_size:-batch_size]], dim=0)

        transition_features = self.to_transition_feature(torch.cat([prev_frames, frame_features], dim=1))
        action_features = self.action_to_feat(torch.cat(actions[:-1], dim=0))
        action_predictions = self.action_discriminator(torch.cat([action_features, transition_features], dim=1))
        if neg_actions is not None:
            neg_action_features = self.action_to_feat(torch.cat(neg_actions[:-1], dim=0))
            neg_action_predictions = self.action_discriminator(
                    torch.cat([neg_action_features, transition_features], dim=1))
        action_z_recon = self.reconstruct_action_z(transition_features)  # frame_features - prev_frames)
        action_recon = action_z_recon[:, :self.opts.action_space]
        z_recon = action_z_recon[:, self.opts.action_space:self.opts.action_space+self.opts.z]

        # run temporal discriminator
        if self.opts.temporal_hierarchy and self.opts.num_steps > 4:
            new_l = []
            temporal_predictions = []
            for entry in tmp_features[:warm_up * batch_size].split(batch_size, dim=0):
                new_l.append(entry)
            for entry in tmp_features[(warm_up + warm_up - 1) * batch_size:].split(batch_size, dim=0):
                new_l.append(entry)
            window_size = len(new_l) #self.temporal_window
            start = np.random.randint(0, len(new_l) - window_size + 1)
            stacked = torch.stack(new_l[start:start + window_size], dim=2)

            aa = self.conv3d[0](stacked)
            a_out = self.conv3d_final[0](aa)
            temporal_predictions.append(a_out.view(batch_size, -1))
            if self.temporal_window >= 12 and epoch >= self.opts.temporal_hierarchy_epoch:
                bb = self.conv3d[1](aa)
                b_out = self.conv3d_final[1](bb)
                temporal_predictions.append(b_out.view(batch_size, -1))
            if self.temporal_window >= 18 and epoch >= self.opts.temporal_hierarchy_epoch:
                cc = self.conv3d[2](bb)
                c_out = self.conv3d_final[2](cc)
                temporal_predictions.append(c_out.view(batch_size, -1))

        dout = {}
        dout['disc_features'] = frame_features[:(len(states)-1)*batch_size]
        dout['action_predictions'] = action_predictions
        dout['single_frame_predictions_all'] = single_frame_predictions_all
        dout['content_predictions'] = temporal_predictions
        dout['neg_action_predictions'] = neg_action_predictions
        dout['action_recon'] = action_recon
        dout['z_recon'] = z_recon
        dout['single_frame_predictions_patch'] = single_frame_predictions_patch
        return dout

    def update_opts(self, opts):
        self.opts = opts
        return
