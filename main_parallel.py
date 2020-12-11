"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import os
import sys
import torch
import time
sys.path.append('..')
import config
import utils
from trainer import Trainer
import torchvision.utils as vutils
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import torch.nn as nn
sys.path.insert(0, './data')
import dataloader
import copy


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(seed)

def train_gamegan(gpu, opts):
    torch.backends.cudnn.benchmark = True

    normalize = True
    opts = copy.deepcopy(opts)
    start_epoch = 0
    opts.img_size = (opts.img_size, opts.img_size)
    warm_up = opts.warm_up
    opts.gpu = gpu
    opts.num_data_types = len(opts.data.split('-'))

    load_weights = False
    # load model
    if opts.saved_model is not None and opts.saved_model != '':
        gpu = opts.gpu
        log_dir = opts.log_dir

        saved_model = torch.load(opts.saved_model, map_location='cpu')
        saved_optim = torch.load(opts.saved_optim, map_location='cpu')
        opts = saved_model['opts']
        opts.gpu = gpu
        opts.log_dir = log_dir
        warm_up = opts.warm_up
        start_epoch = saved_model['epoch']
        load_weights = True

    if opts.num_gpu > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=opts.num_gpu,
            rank=gpu
        )

    torch.manual_seed(opts.seed)
    torch.cuda.set_device(gpu)

    # create model
    netG, netD = utils.build_models(opts)
    # choose optimizer
    optD = utils.choose_optimizer(netD, opts, opts.lrD)
    keyword = 'graphic'
    optG_temporal = utils.choose_optimizer(netG, opts, opts.lrG_temporal, exclude=keyword,
                                               model_name='optG_temporal')
    optG_graphic = utils.choose_optimizer(netG, opts, opts.lrG_graphic, include=keyword, model_name='optG_graphic')
    if load_weights:
        utils.load_my_state_dict(netG, saved_model['netG'])
        utils.load_my_state_dict(netD, saved_model['netD'])
        optG_temporal.load_state_dict(saved_optim['optG_temporal'])
        optG_graphic.load_state_dict(saved_optim['optG_graphic'])
        optD.load_state_dict(saved_optim['optD'])
        del saved_model, saved_optim

    if opts.num_gpu > 1:
        netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu], find_unused_parameters=True)
        netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu], find_unused_parameters=True)


    # dataset ---
    print('setting up dataset')

    if opts.num_gpu > 1:
        train_dataset = dataloader.get_custom_dataset(opts, set_type=0, getLoader=False)
        val_dataset = dataloader.get_custom_dataset(opts, set_type=1, getLoader=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opts.num_gpu,
            shuffle=True,
            rank=gpu
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opts.bs,
            shuffle=False,
            num_workers=5,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=opts.num_gpu,
            shuffle=False,
            rank=gpu
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=opts.bs,
            shuffle=False,
            num_workers=5,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=True)
    else:
        train_loader = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True)
        val_loader = dataloader.get_custom_dataset(opts, set_type=1, getLoader=True)

    # set up logger and trainer
    logging = True if gpu == 0 else False
    if logging:
        logger = SummaryWriter(opts.log_dir)

    zdist = utils.get_zdist('gaussian', opts.z)
    trainer = Trainer(opts,
                      netG, netD,
                      optG_temporal, optG_graphic, optD,
                      opts.gan_type, opts.reg_type, opts.LAMBDA, zdist)

    vis_num_row = 1
    if opts.num_steps > 29:
        vis_num_row = 3
    num_vis = 1
    cur_lr = opts.lr
    for epoch in range(start_epoch, opts.nep):
        if epoch % opts.lr_decay_epoch  == 0 and epoch > 0 and cur_lr > opts.min_lr:
            cur_lr = cur_lr * 0.5
            utils.adjust_learning_rate(optG_temporal, cur_lr)
            utils.adjust_learning_rate(optG_graphic, cur_lr)
            utils.adjust_learning_rate(optD, cur_lr)
        print('Start epoch %d...' % epoch) if logging else None

        data_iters, train_len = [], 99999999999
        data_iters.append(iter(train_loader))
        if len(data_iters[-1]) < train_len:
            train_len = len(data_iters[-1])
        torch.cuda.empty_cache()

        log_iter = max(1,int(train_len // 10))
        write_d = 0

        for step in range(train_len):
            it = epoch * train_len + step

            # prepare data
            sample = None
            states, actions, neg_actions = utils.get_data(data_iters, opts)

            # Generators updates
            start = time.time()
            gloss_dict, gloss, gout, grads, dout_fake = \
                trainer.generator_trainstep(states, actions, warm_up=warm_up, epoch=epoch)
            gtime = time.time() - start

            # Discriminator updates
            if ((it + 1) % opts.Diters) == 0 and opts.gan_loss:
                start = time.time()
                dloss_dict = trainer.discriminator_trainstep(states, actions,
                                                            neg_actions, warm_up=warm_up, gout=gout, dout_fake=dout_fake,
                                                            epoch=epoch, step=step)
                dtime = time.time() - start

            # Log
            if logging:
                with torch.no_grad():
                    if step == 0:
                        utils.plot_grad({'netG': trainer.netG, 'netD': trainer.netD}, logger, it)

                    loss_str = 'Generator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
                    for k, v in gloss_dict.items():
                        if not (type(v) is float):
                            if (step % log_iter) == 0:
                                logger.add_scalar('losses/' + k, v.data.item(), it)
                            loss_str += k + ': ' + str(v.data.item()) + ', '
                    print(loss_str)
                    print('netG update:%f' % (gtime))

                if (step % log_iter) == 0:
                    # logging visualization
                    utils.draw_output(gout, states, warm_up, opts, vutils, vis_num_row, normalize, logger,
                                      it,
                                      num_vis, tag='trn_images')

                if ((it + 1) % opts.Diters) == 0 and opts.gan_loss:
                    loss_str = 'Discriminator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
                    for k, v in dloss_dict.items():
                        if not type(v) is float:
                            if (write_d % (log_iter // opts.Diters) == 0):
                                logger.add_scalar('losses/' + k, v.data.item(), it)
                            loss_str += k + ': ' + str(v.data.item()) + ', '
                    write_d += 1
                    print(loss_str)
                    print('netD update:%f' % (dtime))
            del gloss_dict, gloss, gout, grads, dout_fake, states, actions, neg_actions, sample
            if opts.gan_loss:
                del dloss_dict

        print('Validation epoch %d...' % epoch) if logging else None
        data_iters, val_len = [], 99999999999
        data_iters.append(iter(val_loader))
        if len(data_iters[-1]) < val_len:
            val_len = len(data_iters[-1])
        torch.cuda.empty_cache()

        max_vis = 10
        for step in range(val_len):
            it = epoch * val_len + step

            # prepare data
            states, actions, neg_actions = utils.get_data(data_iters, opts)

            trainer.netG.eval()
            if step < max_vis:
                with torch.no_grad():
                    loss_dict, gloss, gout, _, _ = trainer.generator_trainstep(states, actions, warm_up=warm_up,
                                                                               train=False,
                                                                               epoch=epoch,
                                                                               )
                    if logging:
                        if opts.final_l1 or opts.final_l2:
                            logger.add_scalar('val_losses/recon_loss', loss_dict['loss_recon'], it)
                        utils.draw_output(gout, states, warm_up, opts, vutils, vis_num_row, normalize, logger, it,
                                          num_vis, tag='val_images')
                del loss_dict, gloss, gout
            else:
                break

        save_epoch = opts.save_epoch

        if epoch % save_epoch == 0 and epoch > save_epoch - 1 and logging:
            print('Saving checkpoint')
            utils.save_model(os.path.join(opts.log_dir, 'model' + str(epoch) + '.pt'), epoch, netG, netD, opts)
            utils.save_optim(os.path.join(opts.log_dir, 'optim' + str(epoch) + '.pt'), epoch, optG_temporal,
                             optG_graphic, optD)



if __name__ == '__main__':
    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)
    if opts.num_gpu > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(train_gamegan, nprocs=opts.num_gpu, args=(opts,))
    else:
        train_gamegan(opts.gpu, opts)
