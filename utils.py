"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch import distributions
import math

def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gaussian':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist

def save_model(fname, epoch, netG, netD, opts):
    outdict = {'epoch': epoch, 'netG': netG.state_dict(), 'netD': netD.state_dict(), 'opts': opts}
    torch.save(outdict, fname)

def save_optim(fname, epoch, optG_temporal, optG_graphic, optD):
    outdict = {'epoch': epoch, 'optG_temporal': optG_temporal.state_dict(), 'optG_graphic': optG_graphic.state_dict(), 'optD': optD.state_dict()}
    torch.save(outdict, fname)

def adjust_learning_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def choose_optimizer(model, options, lr=None, exclude=None, include=None, model_name=''):
    try:
        wd = options.wd
    except:
        wd = 0.0

    if lr == None:
        lr = options.lr

    if type(model) is list:
        params = model
    else:
        params = model.parameters()
        if exclude is not None:
            params = []
            for name, W in model.named_parameters():
                if not exclude in name:
                    params.append(W)
                    print(model_name + ', Include: ' + name)
                else:
                    print(model_name + ', Exclude: ' + name)
        if include is not None:
            params = []
            for name, W in model.named_parameters():
                if include in name:
                    params.append(W)
                    print(model_name + ', Include: ' + name)

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=(0.0, 0.9))

    return optimizer



def build_models(opts, tmp_get_old=False):
    from simulator_model.dynamics_engine import EngineGenerator as Generator
    from simulator_model.discriminator import Discriminator

    # Build models
    generator = Generator(
        opts
    )
    discriminator = Discriminator(
        opts,
        nfilter=opts.nfilterD
    )
    if opts.gpu is not None and not opts.gpu < 0 :
        return generator.cuda(opts.gpu), discriminator.cuda(opts.gpu)
    else:
        return generator, discriminator

def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def copy_weights(source, target):
    target.data = source.data
    return

def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in own_state.items():
        print(name)
    for name, param in state_dict.items():
        print(name)
        if name not in own_state:
            name = name.replace('module.', '')
            if name not in own_state:
                continue
        print(name)
        if isinstance(param, nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            print(param.size())
            print(own_state[name].size())
            continue

def plot_grad(ml, logger, step):
    for key, model in ml.items():
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            if value.grad is None:
                print('@@@@@@@@@@@@@' + key + '/' + tag + ' has no grad.')
            else:
               logger.add_histogram('grad/'+key+'/'+tag, value.grad, step)

def check_arg(opts, arg):
    v = vars(opts)
    if arg in v:
        if type(v[arg]) == bool:
            return v[arg]
        else:
            return True
    else:
        return False

def check_gpu(gpu, *args):
    '''
    '''
    if gpu == None or gpu < 0:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key])
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a) for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a) for a in args]
        else:
            return Variable(args[0])

    else:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key]).cuda(gpu)
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a).cuda(gpu) for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a).cuda(gpu) for a in args]
        else:
            return Variable(args[0].cuda(gpu))

def rescale(x):
    return (x + 1) * 0.5

def get_data(data_iters, opts, get_rand=False):

    tmp_states, tmp_actions, tmp_neg_actions, sample = [], [], [], None
    states, actions, neg_actions = [], [], []
    for data_iter in data_iters:
        s, a, na = data_iter.next()
        tmp_states.append(s)
        tmp_actions.append(a)
        tmp_neg_actions.append(na)
    for j in range(len(tmp_states[0])): # over time steps
        gs, ga, gna = [], [], []
        for k in range(len(tmp_states[0][0])): # over batches
            for i in range(len(tmp_states)): # over data type
                gs.append(tmp_states[i][j][k])
                ga.append(tmp_actions[i][j][k])
                gna.append(tmp_neg_actions[i][j][k])
        states.append(torch.stack(gs, dim=0))
        actions.append(torch.stack(ga, dim=0))
        neg_actions.append(torch.stack(gna, dim=0))

    num_data_types = len(tmp_states)

    states = [check_gpu(opts.gpu, a) for a in states]
    actions = [check_gpu(opts.gpu, a) for a in actions]
    neg_actions = [check_gpu(opts.gpu, a) for a in neg_actions]

    return states, actions, neg_actions

def load_state_dict(self, state_dict):
    import torch.nn as nn
    own_state = self.state_dict()
    for name, param in state_dict.items():

        if name not in own_state:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            continue

        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print('++++++++++++++++++++++++++++++ ' + name + ' LOADED')
        except:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            print(param.size())
            print(own_state[name].size())
            continue

def compute_grad2(d_out, x_in, allow_unused=False, batch_size=None, gpu=0, ns=1):
    # Reference:
    # https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    if d_out is None:
        return utils.check_gpu(gpu, torch.FloatTensor([0]))
    if batch_size is None:
        batch_size = x_in.size(0)

    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
        allow_unused=allow_unused
    )[0]

    grad_dout2 = grad_dout.pow(2)
    reg = grad_dout2.view(batch_size, -1).sum(1) * (ns * 1.0 / 6)
    return reg

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def draw_output(gout, states, warm_up, opts, vutils, vis_num_row, normalize, logger, it, num_vis, tag='images'):
    img_size = opts.img_size
    bs, _, h, w = states[0].size()

    if warm_up > 0:
        warm_up_states = torch.cat(states[:warm_up], dim=1)
        warm_up_states = warm_up_states[0:num_vis].view(warm_up * num_vis, opts.num_channel, h, w)
        if opts.penultimate_tanh:
            warm_up_states = rescale(warm_up_states)
        warm_up_states = torch.clamp(warm_up_states, 0, 1.0)
        x = vutils.make_grid(
            warm_up_states, nrow=(warm_up) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_output/WARMUPImage', x, it)

    states_ = torch.cat(states[warm_up:], dim=1)
    states_ = states_[0:num_vis].view((opts.num_steps - warm_up) * num_vis, opts.num_channel, h, w)
    if opts.penultimate_tanh:
        states_ = rescale(states_)
    states_ = torch.clamp(states_, 0, 1.0)
    x = vutils.make_grid(
        states_, nrow=(opts.num_steps - warm_up) // vis_num_row,
        normalize=normalize, scale_each=normalize
    )
    logger.add_image(tag + '_output/GTImage', x, it)


    x_gen = gout['outputs']
    x_gen = torch.cat(x_gen, dim=1)
    x_gen = x_gen[0:num_vis].view(len(gout['outputs']) * num_vis, opts.num_channel, h, w)
    if opts.penultimate_tanh:
        x_gen = rescale(x_gen)
    x_gen = torch.clamp(x_gen, 0, 1.0)
    x = vutils.make_grid(
        x_gen, nrow=len(gout['outputs']) // vis_num_row,
        normalize=normalize, scale_each=normalize
    )
    logger.add_image(tag + '_output/GenImage', x, it)



    mem_h = int(math.sqrt(opts.memory_h))
    mem_w = opts.memory_h // mem_h

    if 'rev_outputs' in gout and len(gout['rev_outputs']) > 0:

        x_rev = torch.cat(gout['rev_inputs'], dim=1)
        x_rev = x_rev[0:num_vis].view(len(gout['rev_inputs']) * num_vis, opts.num_channel, h, w)
        # x_rev = torch.clamp(x_rev, 0, 1.0)
        if opts.penultimate_tanh:
            x_rev = rescale(x_rev)
        x = vutils.make_grid(
            x_rev, nrow=len(gout['rev_inputs']) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_rev_output/RevInputImage', x, it)

        x_rev = torch.cat(gout['rev_outputs'], dim=1)
        x_rev = x_rev[0:num_vis].view(len(gout['rev_outputs']) * num_vis, opts.num_channel, h, w)
        # x_rev = torch.clamp(x_rev, 0, 1.0)
        if opts.penultimate_tanh:
            x_rev = rescale(x_rev)
        x = vutils.make_grid(
            x_rev, nrow=len(gout['rev_outputs']) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_rev_output/RevOutputImage', x, it)

        if opts.do_memory:
            rev_alpha = torch.clamp(torch.cat(gout['rev_alphas'], dim=1), 0, 1.0)
            rev_alpha = rev_alpha[0:num_vis].view(len(gout['rev_alphas']) * num_vis, 1, mem_w, mem_h)
            x = vutils.make_grid(
                rev_alpha, nrow=len(gout['rev_alphas']) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_rev_memory/rev_alphas', x, it)
            if 'sec_rev_alphas' in gout and len(gout['sec_rev_alphas']) > 0:
                rev_alpha = torch.clamp(torch.cat(gout['sec_rev_alphas'], dim=1), 0, 1.0)
                rev_alpha = rev_alpha[0:num_vis].view(len(gout['sec_rev_alphas']) * num_vis, 1, mem_w,
                                                      mem_h)
                x = vutils.make_grid(
                    rev_alpha, nrow=len(gout['sec_rev_alphas']), normalize=False, scale_each=False
                )
                logger.add_image(tag + '_rev_memory/sec_rev_alphas', x, it)

    if opts.do_memory:
        alpha = torch.clamp(torch.cat(gout['alphas'], dim=1), 0, 1.0)
        alpha = alpha[0:num_vis].view(len(gout['alphas']) * num_vis, 1, mem_w, mem_h)
        x = vutils.make_grid(
            alpha, nrow=len(gout['alphas']) // vis_num_row, normalize=False, scale_each=False
        )
        logger.add_image(tag + '_memory/alphas', x, it)

        # import pdb; pdb.set_trace();
        if 'kernels' in gout:
            kernels = torch.clamp(torch.cat(gout['kernels'], dim=1), 0, 1.0)
            kernels = kernels[0:num_vis].view(len(gout['kernels']) * num_vis, 1, mem_w, mem_h)
            x = vutils.make_grid(
                kernels, nrow=len(gout['kernels']) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_memory/kernels', x, it)

    maps = gout['maps']

    if len(maps) > 0:
        for cur_component in range(len(gout['base_imgs_all'][0])):
            gather_recon_maps = []
            len_episode = len(gout['base_imgs_all'])
            for cur_step in range(len_episode):
                gather_recon_maps.append(
                    F.interpolate(gout['base_imgs_all'][cur_step][cur_component], size=img_size,
                                  mode='bilinear'))

            gather_recon_maps = torch.cat(gather_recon_maps, dim=1)

            gather_recon_maps = gather_recon_maps[0:num_vis].view(len_episode * num_vis, opts.num_channel,
                                                                  img_size[0], img_size[1])
            x = vutils.make_grid(
                gather_recon_maps, nrow=len_episode // vis_num_row, normalize=normalize,
                scale_each=normalize
            )
            logger.add_image(tag + '_graphics/recon_x_map' + str(cur_component), x, it)
            if len(gout['rev_outputs']) > 0:
                gather_recon_maps = []
                len_episode = len(gout['rev_base_imgs_all'])
                for cur_step in range(len_episode):
                    gather_recon_maps.append(
                        F.interpolate(gout['rev_base_imgs_all'][cur_step][cur_component], size=img_size,
                                      mode='bilinear'))

                gather_recon_maps = torch.cat(gather_recon_maps, dim=1)
                gather_recon_maps = gather_recon_maps[0:num_vis].view(len_episode * num_vis,
                                                                      opts.num_channel,
                                                                      img_size[0], img_size[1])
                x = vutils.make_grid(
                    gather_recon_maps, nrow=len_episode // vis_num_row, normalize=normalize,
                    scale_each=normalize
                )
                logger.add_image(tag + '_rev_graphics/recon_x_map' + str(cur_component), x, it)

        for cur_component in range(len(maps[0])):
            if len(maps[0]) == 0:
                break
            gather_maps = []
            for cur_step in range(len(maps)):
                gather_maps.append(maps[cur_step][cur_component])

            gather_maps = torch.cat(gather_maps, dim=1)
            gather_maps = gather_maps[0:num_vis].view(len(maps) * num_vis, 1, gather_maps.size(2),
                                                      gather_maps.size(3))
            x = vutils.make_grid(
                gather_maps, nrow=len(maps) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_graphics/Map' + str(cur_component), x, it)

            if 'init_maps' in gout:
                gather_maps = []
                init_maps = gout['init_maps']
                if len(init_maps)> 0 and len(init_maps[0]) > 0 and len(init_maps[0][0]) > 0:
                    for cur_step in range(len(init_maps)):

                        gather_maps.append(init_maps[cur_step][cur_component])

                    gather_maps = torch.cat(gather_maps, dim=1)
                    gather_maps = gather_maps[0:num_vis].view(len(init_maps) * num_vis, 1, gather_maps.size(2),
                                                              gather_maps.size(3))
                    x = vutils.make_grid(
                        gather_maps, nrow=len(init_maps) // vis_num_row, normalize=False, scale_each=False
                    )
                    logger.add_image(tag + '_graphics/init_Map' + str(cur_component), x, it)

            if len(gout['rev_outputs']) > 0:
                gather_maps = []
                if len(gout['rev_maps']) > 0 and len(gout['rev_maps'][0]) > 0 and len(gout['rev_maps'][0][0]) > 0:
                    for cur_step in range(len(gout['rev_maps'])):
                        gather_maps.append(gout['rev_maps'][cur_step][cur_component])

                    gather_maps = torch.cat(gather_maps, dim=1)
                    gather_maps = gather_maps[0:num_vis].view(len(gout['rev_maps']) * num_vis, 1,
                                                              gather_maps.size(2),
                                                              gather_maps.size(3))
                    x = vutils.make_grid(
                        gather_maps, nrow=len(gout['rev_maps']) // vis_num_row, normalize=False,
                        scale_each=False
                    )
                    logger.add_image(tag + '_rev_graphics/Map' + str(cur_component), x, it)
