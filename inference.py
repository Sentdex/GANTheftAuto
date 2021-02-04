"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import os
import sys
import torch
sys.path.append('..')
import config
import utils
import random
import torch.multiprocessing as mp
sys.path.insert(0, './data')
import dataloader
import copy
import cv2
import numpy as np


# Workaround for PyTorch issue on Windows
if os.name == 'nt':
    import ctypes
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


def inference(gpu, opts):

    # Initialize values
    opts = copy.deepcopy(opts)
    opts.img_size = (opts.img_size, opts.img_size)
    warm_up = opts.warm_up
    opts.gpu = gpu
    opts.num_data_types = len(opts.data.split('-'))

    log_dir = opts.log_dir

    # Load teh model
    saved_model = torch.load(opts.saved_model, map_location='cpu')
    #saved_optim = torch.load(opts.saved_optim, map_location='cpu')
    opts = saved_model['opts']
    opts.gpu = gpu
    opts.log_dir = log_dir
    warm_up = opts.warm_up

    # Initialize torch
    torch.manual_seed(opts.seed)
    torch.cuda.set_device(gpu)

    # Create the generator model and load it;s state from the checkpoint
    netG, _ = utils.build_models(opts)
    utils.load_my_state_dict(netG, saved_model['netG'])

    # Initialize the noise generator
    zdist = utils.get_zdist('gaussian', opts.z)

    # For "playing", we want teh batch size of 1
    opts.bs = 1
    batch_size = opts.bs

    # Load the dataset so we can get some initial image
    ##!! Replace with some examle set-aside images
    train_loader = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True)
    data_iters, train_len = [], 99999999999
    data_iters.append(iter(train_loader))
    if len(data_iters[-1]) < train_len:
        train_len = len(data_iters[-1])
    states, actions, neg_actions = utils.get_data(data_iters, opts)

    # Disable gradients, we'll perform just the inference
    utils.toggle_grad(netG, False)
    netG.eval()

    ##!! Temporary actions, replace with actual keys later
    gen_actions = [torch.tensor([np.eye(2)[random.randint(0, 1)]], dtype=torch.float32).cuda() for _ in range(1600)]

    # Disable warmup
    warm_up = 0

    # Temporary, to save predictions as videos
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #v = cv2.VideoWriter('video.mp4', fourcc, 30.0, (600,400))

    # Run warmup to get initial values
    # warmup is set to 0, so initial image is going to be used as input
    prev_state, warm_up_state, M, prev_read_v, prev_alpha, outputs, maps, alphas, alpha_losses, zs, base_imgs_all, _, \
        hiddens, init_maps = netG.run_warmup(zdist, states, actions, warm_up, train=False)
    h, c = warm_up_state

    # Show the image
    img = prev_state[0].cpu().numpy()
    img = np.rollaxis(img, 0, 3)
    img = cv2.resize(img, (400, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('test', img[...,::-1])
    cv2.waitKey(0)

    # Uncomment to wite to the video stream
    #for _ in range(30):
    #    v.write((img*255).astype(np.uint8))

    # Generate 160 frames
    for i in range(160):

        # Perform inference
        prev_state, m, prev_alpha, alpha_loss, z, M, prev_read_v, h, c, init_map, base_imgs, _, cur_hidden = netG.run_step(prev_state, h, c, gen_actions[i], \
                                                                              batch_size, prev_read_v, prev_alpha, M, zdist, step=i)

        # Show the image
        img = prev_state[0].cpu().numpy()
        img = np.rollaxis(img, 0, 3)
        img = cv2.resize(img, (400, 600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('test', img[...,::-1])
        cv2.waitKey(1)

        # Uncomment to wite to the video stream
        #v.write((img*255).astype(np.uint8))


if __name__ == '__main__':
    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)
    inference(opts.gpu, opts)
