"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import math
import utils
import sys
sys.path.append('..')

from simulator_model import model_utils
from simulator_model.rendering_engine import RenderingEngine
from simulator_model.memory import Memory


class ActionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=1024, opts=None):
        super(ActionLSTM, self).__init__()
        self.opts = opts
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.v_size = hidden_size
        self.project_input = nn.Linear(self.input_dim, self.input_dim)
        self.project_h = nn.Linear(self.hidden_size, self.input_dim)

        self.v2h = nn.Sequential(nn.Linear(self.input_dim, self.input_dim),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(self.input_dim, 4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_hidden(self, bs):
        '''
        initialize the hidden states to be 0
        '''
        return torch.zeros(bs, self.hidden_size), torch.zeros(bs, self.hidden_size)

    def forward(self, h, c, input, state_bias=None, step=0):
        """
        :param h: prev hidden
        :param c: prev cell
        :param input: input
        :return:
        """
        h_proj = self.project_h(h)

        input_proj = self.project_input(input)
        v = self.v2h(h_proj * input_proj)


        if state_bias is None:
            state_bias = 0
        tmp = v + state_bias

        # activations
        g_t = tmp[:, 2 * self.hidden_size:3 * self.hidden_size].tanh()
        i_t = tmp[:, :self.hidden_size].sigmoid()
        f_t = tmp[:, self.hidden_size:2 * self.hidden_size].sigmoid()
        o_t = tmp[:, -self.hidden_size:].sigmoid()

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class EngineModule(nn.Module):
    def __init__(self, opts, state_dim):
        super(EngineModule, self).__init__()
        self.hdim = opts.hidden_dim
        self.opts = opts

        action_space = 10 if not utils.check_arg(self.opts, 'action_space') else self.opts.action_space

        self.a_to_input = nn.Sequential(nn.Linear(action_space, self.hdim),
                                        nn.LeakyReLU(0.2))
        self.z_to_input = nn.Sequential(nn.Linear(opts.z, self.hdim),
                                        nn.LeakyReLU(0.2))
        # engine module input network
        e_input_dim = self.hdim*2
        if self.opts.do_memory:
            e_input_dim += self.opts.memory_dim

        self.f_e = nn.Sequential(nn.Linear(e_input_dim, e_input_dim),
                                 nn.LeakyReLU(0.2))

        self.rnn_e = ActionLSTM(e_input_dim,
                                hidden_size=self.hdim, opts=opts)

        self.state_bias = nn.Sequential(nn.Linear(state_dim, self.hdim * 4))


    def init_hidden(self, bs):
        '''
        initialize the hidden states to be 0
        '''
        return [torch.zeros(bs, self.rnn_e.hidden_size)], [torch.zeros(bs, self.rnn_e.hidden_size)]

    def forward(self, h, c, s, a, z, prev_read_v=None, step=0):
        h_e = h[0]
        c_e = c[0]

        # prepare inputs
        if self.opts.do_memory:
            input_core = [self.a_to_input(a), self.z_to_input(z), prev_read_v]
        else:
            input_core = [self.a_to_input(a), self.z_to_input(z)]
        state_bias = self.state_bias(s)
        input_core = torch.cat(input_core, dim=1)

        # Core engine
        e = self.f_e(input_core)
        h_e_t, c_e_t = self.rnn_e(h_e, c_e, e, state_bias=state_bias, step=step)

        H = [h_e_t]
        C = [c_e_t]

        return H, C, h_e_t


class EngineGenerator(nn.Module):

    def __init__(self, opts, nfilter_max=512, **kwargs):
        super(EngineGenerator, self).__init__()

        self.opts = opts

        self.z_dim = opts.z
        self.num_components = opts.num_components
        self.hdim = opts.hidden_dim
        self.expand_dim = opts.nfilterG

        if opts.do_memory:
            self.base_dim = opts.memory_dim
        else:
            self.base_dim = opts.hidden_dim
        state_dim_multiplier = 1
        if utils.check_arg(self.opts, 'state_dim_multiplier'):
            state_dim_multiplier = self.opts.state_dim_multiplier
        state_dim = self.base_dim * state_dim_multiplier

        # Memory Module
        if self.opts.do_memory:
            self.memory = Memory(opts, state_dim)

        # Dynamics Engine
        self.engine = EngineModule(opts, state_dim)
        self.simple_enc = model_utils.choose_netG_encoder(basechannel=state_dim, opts=self.opts)

        # Rendering Engine
        self.graphics_renderer = RenderingEngine(G_ch=opts.nfilterG, opts=opts, resolution=self.opts.img_size[0])

        self.num_components = self.opts.num_components



    def run_step(self, state, h, c, action, batch_size, prev_read_v, prev_alpha, M, zdist,
                read_only=False, step=0, decode=True, play=False, force_sharp=False):
        '''
        Run the model one time step
        '''

        # encode the image input
        if self.opts.input_detach:
            state = state.detach()
        s = self.simple_enc(state)

        # sample a noise
        z = utils.check_gpu(self.opts.gpu, zdist.sample((batch_size,)))

        # run dynamics engine
        prev_hidden = h[0].clone()
        h, c, cur_hidden = self.engine(h, c, s, action, z, prev_read_v=prev_read_v, step=step)

        # run memory module
        if self.opts.do_memory:
            base, M, alpha, prev_read_v = self.memory(cur_hidden, action, prev_hidden, prev_alpha, M, c=c[0], read_only=read_only, force_sharp=force_sharp)
            prev_alpha = alpha
            bases = base
        else:
            base = cur_hidden
            bases = [base] * self.num_components

        # run the rendering engine
        alpha_loss = 0
        out, m, eloss, init_maps, base_imgs = self.graphics_renderer(bases, num_components=self.num_components)
        if utils.check_arg(self.opts, 'alpha_loss_multiplier') and self.opts.alpha_loss_multiplier > 0:
            # memory regularization
            for i in range(1, len(m)):
                alpha_loss += (m[i].abs().sum() / batch_size)

        prev_state = out
        return prev_state, m, prev_alpha, alpha_loss, z, M, prev_read_v, h, c, init_maps, base_imgs, 0, cur_hidden

    def run_warmup(self, zdist, states, actions, warm_up, train=True,  M=None,
                   prev_alpha=None, prev_read_v=None, force_sharp=False):
        '''
        Run warm-up phase
        '''
        batch_size = states[0].size(0)
        prev_state = None
        h, c = self.engine.init_hidden(batch_size)
        h = utils.check_gpu(self.opts.gpu, h)
        c = utils.check_gpu(self.opts.gpu, c)

        outputs, maps, zs, alphas, alpha_logits = [], [], [], [], []
        init_maps = []

        if utils.check_arg(self.opts, 'do_memory'):
            # initialize memory and alpha
            if M is None:
                M = self.memory.init_memory(batch_size)
            if prev_alpha is None:
                prev_alpha = utils.check_gpu(self.opts.gpu, torch.zeros(batch_size, self.memory.num_mem))
                mem_wh = int(math.sqrt(prev_alpha.size(1)))
                prev_alpha[:, mem_wh * (mem_wh // 2) + mem_wh // 2] = 1.0
            if prev_read_v is None:
                prev_read_v = utils.check_gpu(self.opts.gpu, torch.zeros(batch_size, self.opts.memory_dim))
        alpha_losses = 0
        base_imgs_all = []
        hiddens = []
        for i in range(warm_up):
            input_state = states[i]
            prev_state, m, prev_alpha, alpha_loss, z, M, prev_read_v, h, c, init_map, base_imgs, _, cur_hidden = self.run_step(
                input_state, h, c, actions[i], \
                batch_size, prev_read_v, prev_alpha, M, zdist, step=i, force_sharp=force_sharp)
            outputs.append(prev_state)
            maps.append(m)
            alphas.append(prev_alpha)
            alpha_losses += alpha_loss
            zs.append(z)
            base_imgs_all.append(base_imgs)
            hiddens.append(cur_hidden)
            init_maps.append(init_map)

        warm_up_state = [h, c]
        if prev_state is None:
            prev_state = states[0]  # warm_up is 0, the initial screen is always used
        return prev_state, warm_up_state, M, prev_read_v, prev_alpha, outputs, maps, alphas, alpha_losses, zs, base_imgs_all, 0, \
               hiddens, init_maps

    def forward(self, zdist, states, actions, warm_up, train=True, epoch=0):
        batch_size = states[0].size(0)

        # if warm_up is not 0, run warm-up phase where ground truth images are used as input
        prev_state, warm_up_state, M, prev_read_v, prev_alpha, outputs, maps, alphas, alpha_losses, zs, base_imgs_all, _, \
            hiddens, init_maps= self.run_warmup(zdist, states, actions, warm_up, train=train)
        h, c = warm_up_state

        rev_outputs, rev_base_imgs, rev_alphas, rev_maps = [], [], [], []
        response = {}
        forward_end = len(actions) - 1
        for i in range(warm_up, forward_end):
            # run for one time step
            prev_state, m, prev_alpha, alpha_loss, z, M, prev_read_v, h, c, init_map, base_imgs, _, cur_hidden = self.run_step(prev_state, h, c, actions[i], \
                                                                                  batch_size, prev_read_v, prev_alpha, M, zdist, step=i)
            outputs.append(prev_state)
            maps.append(m)
            init_maps.append(init_map)
            alphas.append(prev_alpha)
            alpha_losses += alpha_loss
            zs.append(z)
            base_imgs_all.append(base_imgs)
            hiddens.append(cur_hidden)
        alpha_losses = alpha_losses / forward_end


        if self.opts.do_memory and (self.opts.cycle_loss and self.opts.cycle_start_epoch <= epoch):
            # read from previously visited locations for cycle loss
            for i in range(len(alphas) - 1, -1, -1):
                cur_read_v, _ = self.memory.read(alphas[i], M)
                out, m, eloss, _, base_imgs = self.graphics_renderer([cur_read_v, torch.zeros_like(cur_read_v)], get_mem_only=True)
                if self.opts.rev_multiply_map:
                    rev_outputs.append(base_imgs[2] * maps[i][0])
                else:
                    rev_outputs.append(base_imgs[2])
                rev_maps.append(m)
                rev_base_imgs.append(base_imgs)
                rev_alphas.append(alphas[i])

        response['alpha_loss'] = alpha_losses
        response['rev_outputs'] = rev_outputs
        response['rev_alphas'] = rev_alphas
        response['rev_maps'] = rev_maps
        response['rev_base_imgs_all'] = rev_base_imgs
        response['maps'] = maps
        response['init_maps'] = init_maps
        response['zs'] = zs
        response['outputs'] = outputs
        response['alphas'] = alphas
        response['base_imgs_all'] = base_imgs_all
        return response

    def update_opts(self, opts):
        self.opts = opts
        return
