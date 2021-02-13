"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import math
import utils
import sys
sys.path.append('..')

class Memory(nn.Module):
    def __init__(self, opts, state_dim):

        super(Memory, self).__init__()
        self.opts = opts
        self.num_mem = self.opts.memory_h
        self.mem_h = int(math.sqrt(self.num_mem))

        self.use_h = True
        self.init_dim = 0
        if self.use_h:
            self.init_dim += self.opts.hidden_dim

        self.get_vars_h = nn.Sequential(nn.Linear(self.init_dim, self.opts.memory_dim * 3))

        self.get_kernel = nn.Sequential(nn.Linear(self.opts.action_space, self.opts.memory_dim),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(self.opts.memory_dim, 9))
        self.get_gate = nn.Sequential(nn.Linear(self.opts.hidden_dim, self.opts.memory_dim),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(self.opts.memory_dim, 1),
                                      nn.Sigmoid())

        self.register_buffer('mem_bias', torch.Tensor(self.num_mem, self.opts.memory_dim))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(self.num_mem + self.opts.memory_dim))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def init_memory(self, bs):
        memory = self.mem_bias.clone().repeat(bs, 1, 1)
        return utils.check_gpu(self.opts.gpu, memory)

    def read(self, alpha, M):
        v = torch.bmm(alpha.unsqueeze(1), M)
        v = v.squeeze(1)
        cropped = None
        return v, cropped

    def write(self, erase_v, add_v, alpha_write, M):
        erase = torch.bmm(alpha_write.unsqueeze(-1), erase_v.unsqueeze(1))
        add = torch.bmm(alpha_write.unsqueeze(-1), add_v.unsqueeze(1))
        M = M * (1 - erase) + add
        return M

    def conv2d_with_kernel(self, prev_v, kernels, v_dim=1):

        bs = prev_v.size(0)
        hw = prev_v.size(2)
        khw = kernels.size(2)
        tmp_v = prev_v.view(1, bs*v_dim, hw, hw)
        tmp_k = kernels.view(bs, 1, 1, khw, khw).repeat(1, v_dim, 1, 1, 1).view(bs*v_dim, 1, khw, khw)

        v = F.conv2d(tmp_v, tmp_k, padding=kernels.size(2)//2, groups=bs*v_dim)
        v = v.view(bs, v_dim, hw, hw)

        return v

    def forward(self, h, a, prev_h, prev_alpha, M, c=None, read_only=False, play=False, force_sharp=False):
        bs = a.size(0)
        if self.opts.mem_use_h:
            memory_q_input = h
        else:
            h_norm = F.normalize(h, dim=1)
            prev_h_norm = F.normalize(prev_h, dim=1)

            memory_q_input = h_norm - prev_h_norm

        kernels = self.get_kernel(a).view(-1, 1, 3, 3)

        # flipping kernels (e.g. kernel of Left == flipped kernel of Right
        new_a = a.cpu().numpy()
        _, action_label = torch.max(a, 1)
        action_label = action_label.long().cpu().numpy()
        mask = np.zeros((bs, 1))
        for i in range(bs):
            if 'pacman' in self.opts.data:
                if action_label[i] == 2:
                    new_a[i][1] = 1.0
                    new_a[i][2] = 0.0
                    mask[i][0] = 1.0
                elif action_label[i] == 4:
                    new_a[i][3] = 1.0
                    new_a[i][4] = 0.0
                    mask[i][0] = 1.0
            elif 'vizdoom' in self.opts.data:
                if action_label[i] == 0:
                    new_a[i][1] = 1.0
                    new_a[i][0] = 0.0
                    mask[i][0] = 1.0
            elif 'vroom' in self.opts.data:
                if action_label[i] == 1:
                    new_a[i][1] = 1.0
                    new_a[i][0] = 0.0
                    mask[i][0] = 1.0
        mask = utils.check_gpu(self.opts.gpu, torch.FloatTensor(mask)).view(-1, 1, 1, 1)
        new_a = utils.check_gpu(self.opts.gpu, torch.FloatTensor(new_a))

        flipped_kernels = torch.flip(self.get_kernel(new_a).view(-1, 1, 3, 3), [2,3])
        kernels = (1-mask) * kernels + mask * flipped_kernels
        if self.opts.softmax_kernel:
            if force_sharp:
                tmp = torch.zeros_like(kernels.view(bs, -1))
                tmp[0][kernels.view(bs, -1).max(1)[1]] = 1.0
                kernels = tmp
            else:
                kernels = F.softmax(kernels.view(bs, -1)/self.opts.alpha_T, dim=1)
            kernels = kernels.view(bs, 1, 3, 3)

        gate = self.get_gate(memory_q_input)
        if force_sharp:
            if gate[0] > 0.5:
                gate = torch.ones_like(gate)
            else:
                gate = torch.zeros_like(gate)
        mem_h = int(math.sqrt(self.num_mem))
        alpha = self.conv2d_with_kernel(prev_alpha.view(bs, 1, mem_h, mem_h), kernels, v_dim=1)
        alpha = alpha.view(bs, -1)
        alpha = alpha * gate + prev_alpha * (1 - gate)
        if force_sharp:
            tmp = torch.zeros_like(alpha)
            tmp[0][alpha.view(bs, -1).max(1)[1]] = 1.0
            alpha = tmp


        if not read_only:
            tmp = self.get_vars_h(h)
            erase_v = tmp[:, :self.opts.memory_dim]
            add_v = tmp[:, 1*self.opts.memory_dim:2*self.opts.memory_dim]
            other_v = tmp[:, 2*self.opts.memory_dim:]

            erase_v = F.sigmoid(erase_v)
            M = self.write(erase_v, add_v, alpha, M)

        read_v, block_read_v = self.read(alpha, M)
        final_h = [read_v, other_v]

        return final_h, M, alpha, read_v
