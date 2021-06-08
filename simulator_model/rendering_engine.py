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
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
import utils
import sys
from simulator_model.model_utils import SPADE
sys.path.append('..')

from simulator_model.model_utils import View
from simulator_model import layers
import functools
from torch.nn import init

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch['160x96'] = {'in_channels' :  [ch * item for item in [32, 16, 8, 4, 2, 2]],
                      'out_channels' : [ch * item for item in [16, 8, 4, 2, 2, 2]],
                      'upsample' : [2] * 4 + [1] * 2,
                      'resolution' : [16, 32, 64, 128, 256, 512],
                      'attention' : {res: (res in [int(item) for item in attention.split('_')])
                                     for res in [16, 32, 64, 128, 256, 512]}}
    arch['80x48'] = {'in_channels' :  [ch * item for item in [16, 8, 4, 2, 2]],
                     'out_channels' : [ch * item for item in [8, 4, 2, 2, 2]],
                     'upsample' : [2] * 3 + [1] * 2,
                     'resolution' : [16, 32, 64, 128, 256],
                     'attention' : {res: (res in [int(item) for item in attention.split('_')])
                                    for res in [16, 32, 64, 128, 256]}}
    arch['128x128'] = {'in_channels' :  [ch * item for item in [16, 8, 4, 2]],
                       'out_channels' : [ch * item for item in [8, 4, 2, 1]],
                       'upsample' : [True] * 5,
                       'resolution' : [16, 32, 64, 128],
                       'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                      for i in range(4, 8)}}
    arch['84x84'] = {'in_channels': [ch * item for item in [16, 8, 4]],
                     'out_channels': [ch * item for item in [8, 4, 2]],
                     'upsample': [True] * 3,
                     'resolution': [16, 32, 64],
                     'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                   for i in range(4, 7)}}
    arch['64x64'] = {'in_channels': [ch * item for item in [16, 8, 4]],
                     'out_channels': [ch * item for item in [8, 4, 2]],
                     'upsample': [True] * 3,
                     'resolution': [16, 32, 64],
                     'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                   for i in range(4, 7)}}

    return arch

class RenderingEngine(nn.Module):
    def __init__(self, G_ch=64, dim_z=512, bottom_width=(8, 8), resolution=(64, 64),
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn', opts=None,
                 **kwargs):
        super(RenderingEngine, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = opts.hidden_dim if not opts.do_memory else opts.memory_dim
        # The initial spatial dimensions
        if (resolution[0] == 96 and resolution[1] == 160) or (resolution[0] == 48 and resolution[1] == 80):
            bottom_width = (6, 10)
        elif resolution[0] == 84 or utils.check_arg(opts, 'simple_blocks'):
            bottom_width = (7, 7)
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = '64_32' if (resolution[0] == 96 and resolution[1] == 160) or (resolution[0] == 48 and resolution[1] == 80) else G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[f'{resolution[1]}x{resolution[0]}']
        self.opts = opts

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':

            #self.which_conv = functools.partial(layers.SNConv2d,
            #                                    kernel_size=3, padding=1,
            #                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
            #                                    eps=self.SN_eps)
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_conv_ker2 = functools.partial(layers.SNConv2d,
                                                kernel_size=2, padding=0,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.get_map_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=1, padding=0,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding

        # Prepare model
        # First linear layer
        self.linear = []
        self.get_map = []
        self.all_blocks = []
        self.output_layer = []
        self.spade_layers = []
        self.out_to_one_dim = []
        self.repeat = opts.num_components

        self.free_dynamic_component = utils.check_arg(self.opts, 'free_dynamic_component')

        for ind in range(self.repeat):
            if self.repeat > 1:
                in_dim = self.dim_z

                att_dim = self.opts.att_dim

                self.get_map.append(nn.Sequential(
                    nn.Linear(self.dim_z, (1 + att_dim) * (self.bottom_width[0] * self.bottom_width[1])),
                    View((-1, 1 + att_dim, self.bottom_width[0], self.bottom_width[1]))
                ))
                if self.opts.spade_index > -1:
                    if self.opts.spade_index == 0:
                        spade_in_chan = self.opts.fixed_v_dim
                    else:
                        spade_in_chan = self.arch['in_channels'][self.opts.spade_index]
                    self.spade_layers.append(SPADE(spade_in_chan, att_dim))

                if (ind >= 1 and not self.free_dynamic_component) or (ind==0):
                    self.linear.append(nn.Sequential(self.which_linear(in_dim,self.opts.fixed_v_dim)))
                else:
                    self.linear.append(self.which_linear(in_dim,
                                                         self.arch['in_channels'][0] * (self.bottom_width[0] * self.bottom_width[1])))

            else:
                in_dim = self.dim_z
                self.linear.append(self.which_linear(in_dim,
                                                     self.arch['in_channels'][0] * (self.bottom_width[0] * self.bottom_width[1])))
            
            # self.blocks is a doubly-nested list of modules, the outer loop intended
            # to be over blocks at a given resolution (resblocks and/or self-attention)
            # while the inner loop is over a given block
            self.blocks = []
            for index in range(len(self.arch['out_channels'])):
                upsample_factor = 2 if type(self.arch['upsample'][index]) is bool else self.arch['upsample'][index]
                if index == 0:
                    self.in_dim = self.arch['in_channels'][index] if (ind == 1 and self.free_dynamic_component) or (self.repeat < 2) else self.opts.fixed_v_dim
                    in_dim = self.in_dim
                    if resolution[0] == 84:
                        upsample_factor = 3
                else:
                    in_dim = self.arch['in_channels'][index]

                if utils.check_arg(opts, 'simple_blocks'):
                    self.blocks += [[layers.SimpleGBlock(in_channels=in_dim,
                                                   out_channels=self.arch['out_channels'][index],
                                                    num_conv = 2 if index == 0 else 1
                                                   )]]
                else:
                    self.blocks += [[layers.GBlock(in_channels=in_dim,
                                                   out_channels=self.arch['out_channels'][index],
                                                   which_conv=self.which_conv,
                                                   activation=self.activation,
                                                   upsample=(functools.partial(F.interpolate, scale_factor=upsample_factor)
                                                             if self.arch['upsample'][index] else None))]]

                # If attention on this block, attach it to the end
                if self.arch['attention'][self.arch['resolution'][index]] and (self.repeat <= 1 or not utils.check_arg(self.opts, 'no_attention')):
                    if not utils.check_arg(opts, 'simple_blocks'):
                        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

            # Turn self.blocks into a ModuleList so that it's all properly registered.
            self.all_blocks.append(nn.ModuleList([nn.ModuleList(block) for block in self.blocks]))

            # output layer: batchnorm-relu-conv.
            # Consider using a non-spectral conv here
            last_conv = self.which_conv
            if utils.check_arg(self.opts, 'simple_blocks'):
                last_conv = self.which_conv_ker2
            if self.opts.no_in:
                self.output_layer.append(nn.Sequential(self.activation,
                                                       last_conv(self.arch['out_channels'][-1], 3)))
            else:
                self.output_layer.append(nn.Sequential(nn.InstanceNorm2d(self.arch['out_channels'][-1]),
                                                  self.activation,
                                                    last_conv(self.arch['out_channels'][-1], 3)))
            if self.repeat > 1:
                self.out_to_one_dim.append(nn.Sequential(self.activation,
                                            last_conv(self.arch['out_channels'][-1], 1),
                                                         nn.LeakyReLU(0.2)))

        self.linear = nn.ModuleList(self.linear)
        self.all_blocks = nn.ModuleList(self.all_blocks)
        self.output_layer = nn.ModuleList(self.output_layer)

        if self.repeat > 1:
            if self.opts.spade_index > -1:
                self.spade_layers = nn.ModuleList(self.spade_layers)
            self.get_map = nn.ModuleList(self.get_map)
            self.out_to_one_dim = nn.ModuleList(self.out_to_one_dim)
            self.fine_mask = nn.Sequential(nn.Conv2d(opts.num_components, 512, 1, 1),
                                           nn.LeakyReLU(0.2),
                                           nn.Conv2d(512, opts.num_components, 1))

        else:
            if self.opts.do_memory:
                self.out_to_one_dim = nn.Sequential(self.activation,
                                                    self.which_conv(self.arch['out_channels'][-1], 1),
                                                    nn.LeakyReLU(0.2))
                self.fine_mask = nn.Sequential(nn.Conv2d(2, 512, 1, 1),
                                               nn.LeakyReLU(0.2),
                                               nn.Conv2d(512, 2, 1))

        self.base_temperature = self.opts.temperature
        if utils.check_arg(self.opts, 'base_temperature'):
            self.base_temperature = self.opts.base_temperature

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()


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
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z, num_components=1, get_mem_only=False):

        if self.repeat == 1:
            # simple rendering engine
            zs = z[0]
            bs = z[0].size(0)
            h = self.linear[0](zs)
            h = h.view(h.size(0), -1, self.bottom_width[0], self.bottom_width[1])
            for index, blocklist in enumerate(self.all_blocks[0]):
                for block in blocklist:
                    h = block(h)
            hs = h.split(bs, dim=0)

            return torch.tanh(self.output_layer[0](hs[0])), [], 0, [] ,[]
        else:
            # disentangling rendering engine
            #assert(self.opts.do_memory == True)
            bs = z[0].size(0)
            vs, atts, maps = [], [], []

            # Rough sketch stage
            for ind in range(self.repeat):
                cur_v = self.linear[ind](z[ind])
                cur_tmp = self.get_map[ind](z[ind])
                cur_map = cur_tmp[:, 0:1, :, :]
                cur_att = cur_tmp[:, 1:, :, :]
                vs.append(cur_v)
                atts.append(cur_att)
                maps.append(cur_map)
            maps = torch.cat(maps, dim=1)
            if self.opts.sigmoid_maps:
                maps = torch.sigmoid(maps / self.base_temperature)
            else:
                maps = torch.softmax(maps / self.base_temperature, dim=1)
            maps = maps.split(1, dim=1)

            # conv transposed operations & attribute stage
            output_img, base_imgs, base_singles, unmasked_base_imgs = 0, [], [], []
            for ind in range(self.repeat):
                cur_map = maps[ind]

                # put object vectors based on the map locations
                if (not self.free_dynamic_component or (self.opts.do_memory and ind == 0)):
                    cur_v = cur_map * \
                        vs[ind].unsqueeze(-1).unsqueeze(-1).expand(bs, vs[ind].size(1),
                                                                 self.bottom_width[0],
                                                                 self.bottom_width[1])
                else:
                    cur_v = cur_map * vs[ind].view(bs, -1, self.bottom_width[0], self.bottom_width[1])

                h, bind = cur_v, ind
                for index, blocklist in enumerate(self.all_blocks[bind]):
                    do_norm, do_activation = True, True
                    if self.opts.spade_index > -1:
                        if self.opts.no_in and index > self.opts.spade_index:
                            do_norm = False
                        if index == self.opts.spade_index:
                            # use attribues through SPADE layer
                            do_norm = False
                            do_activation = False
                            h = self.spade_layers[bind](h, cur_map * atts[ind])
                    for block in blocklist:
                        h = block(h, do_norm=do_norm, do_activation=do_activation)
                base_singles.append(self.out_to_one_dim[bind](h))
                unmasked_base_imgs.append(torch.tanh(self.output_layer[bind](h)))

            # final rendering stage
            fine_mask = F.softmax(self.fine_mask(torch.cat(base_singles, dim=1)) / self.opts.temperature, dim=1)
            fine_masks = torch.split(fine_mask, 1, dim=1)
            for ind in range(self.repeat):
                masked_img = unmasked_base_imgs[ind] * fine_masks[ind]
                base_imgs.append(masked_img)
                output_img += masked_img
            for ind in range(len(unmasked_base_imgs)):
                base_imgs.append(unmasked_base_imgs[ind])

            return output_img, fine_masks, 0,  maps, base_imgs
