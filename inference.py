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
sys.path.insert(0, './data')
import dataloader
import copy
import cv2
import numpy as np
import keyboard
import time

'''
# Workaround for PyTorch issue on Windows
if os.name == 'nt':
    import ctypes
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
'''

def inference(gpu, opts):

    # Initialize values
    opts = copy.deepcopy(opts)
    opts.img_size = (opts.img_size, opts.img_size)
    warm_up = opts.warm_up
    opts.gpu = gpu
    if opts.data is not None:
        opts.num_data_types = len(opts.data.split('-'))

    log_dir = opts.log_dir

    # Multi-part model?
    if not os.path.exists(opts.saved_model):
        part_list = sorted([file for file in os.listdir(os.path.dirname(opts.saved_model)) if os.path.basename(opts.saved_model) in file])
        if len(part_list):
            with open(opts.saved_model, 'wb') as sf:
                for part in part_list:
                    with open(os.path.dirname(opts.saved_model) + '/' + part, 'rb') as lf:
                        sf.write(lf.read())

    # Load the model
    saved_model = torch.load(opts.saved_model, map_location='cpu')
    opts_data = opts.data
    opts_img = opts.inference_image_path
    base_imgs = opts.show_base_images
    upsample_model = opts.upsample_model
    playback_fps = opts.playback_fps
    opts = saved_model['opts']
    if opts_data is not None:
        opts.data = opts_data
    #if opts_img is not None:
    opts.inference_image_path = opts_img
    opts.show_base_images = base_imgs
    opts.upsample_model = upsample_model
    opts.playback_fps = playback_fps
    opts.gpu = gpu
    if type(opts.img_size) == int:
        opts.img_size = [opts.img_size] * 2
    opts.log_dir = log_dir
    warm_up = opts.warm_up

    curdata, datadir = opts.data.split(':')
    if curdata == 'cartpole':
        resized_image_size = (600, 400)
        action_left = [1, 0]
        action_right = [0, 1]
        no_action = None
    elif curdata == 'vroom':
        resized_image_size = (256, 256)
        action_left = [1, 0, 0]
        action_right = [0, 0, 1]
        no_action = [0, 1, 0]
    elif curdata == 'gtav':
        resized_image_size = (320, 192)
        action_left = [1, 0, 0]
        action_right = [0, 0, 1]
        no_action = [0, 1, 0]
    else:
        raise Exception(f'Not implemented: unknown data type: {curdata}')

    # Initialize torch
    torch.manual_seed(opts.seed)
    
    if torch.cuda.is_available():    
        torch.cuda.set_device(gpu)
    else:
        raise Exception('CUDA is not available. Please check that you have an Nvidia CUDA GPU')

    # Create the generator model and load it;s state from the checkpoint
    netG, _ = utils.build_models(opts)
    utils.load_my_state_dict(netG, saved_model['netG'])

    # Initialize the noise generator
    zdist = utils.get_zdist('gaussian', opts.z)

    # For "playing", we want teh batch size of 1
    opts.bs = 1
    batch_size = opts.bs

    if opts.inference_image_path is None:
        # Load the dataset so we can get some initial image
        ##!! Replace with some examle set-aside images
        train_loader = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True)
        data_iters, train_len = [], 99999999999
        data_iters.append(iter(train_loader))
        if len(data_iters[-1]) < train_len:
            train_len = len(data_iters[-1])
        states, actions, _ = utils.get_data(data_iters, opts)
    else:
        # Load starting image
        img = cv2.imread(opts.inference_image_path)[...,::-1]
        img = (np.transpose(img, axes=(2, 0, 1)) / 255.).astype('float32')
        img = (img - 0.5) / 0.5

        states = [torch.tensor([img], dtype=torch.float32).cuda()]
        actions = [torch.tensor(no_action if no_action is not None else action_left, dtype=torch.float32).cuda()]

    # Disable gradients, we'll perform just the inference
    utils.toggle_grad(netG, False)
    netG.eval()

    # Disable warmup
    warm_up = 0

    # Temporary, to save predictions as videos
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #v = cv2.VideoWriter('video.mp4', fourcc, 10.0, resized_image_size)

    upsample = None
    if opts.upsample_model != None:
        from upsample import upsample
        upsample.load(opts.upsample_model)

    action = None
    hidden_action = 0
    prev_state = None

    i = 0
    while True:

        frame_start_time = time.time()

        action_text = ''
        if keyboard.is_pressed('e'):
            exit()
        elif keyboard.is_pressed('r') or prev_state is None:
            # Run warmup to get initial values
            # warmup is set to 0, so initial image is going to be used as input
            prev_state, warm_up_state, M, prev_read_v, prev_alpha, outputs, maps, alphas, alpha_losses, zs, base_imgs_all, _, \
                hiddens, init_maps = netG.run_warmup(zdist, states, actions, warm_up, train=False)
            h, c = warm_up_state

            # Show the image
            img = prev_state[0].cpu().numpy()
            img = np.rollaxis(img, 0, 3)
            img = ((img+1)*127.5).astype(np.uint8)
            img = cv2.resize(img, resized_image_size, interpolation=cv2.INTER_NEAREST)
            img = img[...,::-1]

            cv2.imshow(f'{curdata} - inference', img)
            if upsample is not None:
                upsampled_img = upsample.inference(np.rollaxis(prev_state[0].cpu().numpy(), 0, 3))
                if type(upsampled_img) is np.ndarray:
                    cv2.imshow(f'{curdata} - upsampled', upsampled_img[0][...,::-1])
                else:
                    cv2.imshow(f'{curdata} - upsampled', upsampled_img[0][0][...,::-1])
                    if len(upsampled_img) > 1:
                        cv2.imshow(f'{curdata} - upsampled aux1', upsampled_img[1][0][...,::-1])
                    if len(upsampled_img) > 2:
                        cv2.imshow(f'{curdata} - upsampled aux2', upsampled_img[2][0][...,::-1])

            cv2.waitKey(1000)

            # Uncomment to wite to the video stream
            #for _ in range(30):
            #    v.write(img)

            continue
        elif keyboard.is_pressed('a'):
            action = torch.tensor([action_left], dtype=torch.float32).cuda()
            hidden_action = -1
            #action_text = 'LEFT'
        elif keyboard.is_pressed('d'):
            action = torch.tensor([action_right], dtype=torch.float32).cuda()
            hidden_action = 1
            #action_text = 'RIGHT'
        elif no_action is not None:
            action = torch.tensor([no_action], dtype=torch.float32).cuda()
            hidden_action = 0
        else:
            action = torch.tensor([np.eye(opts.action_space)[random.randint(0, np.eye(opts.action_space) - 1)]], dtype=torch.float32).cuda()
            hidden_action = None
        #print(action, action_text)

        # Perform inference
        prev_state, m, prev_alpha, alpha_loss, z, M, prev_read_v, h, c, init_map, base_imgs, _, cur_hidden = netG.run_step(prev_state, h, c, action, \
                                                                              batch_size, prev_read_v, prev_alpha, M, zdist, step=i)

        # Show the image
        img = prev_state[0].cpu().numpy()
        img = np.rollaxis(img, 0, 3)
        img = ((img+1)*127.5).astype(np.uint8)
        img = cv2.resize(img, resized_image_size, interpolation=cv2.INTER_NEAREST)
        img = img[...,::-1]
        rectangle = img.copy()
        cv2.rectangle(rectangle, (0, 0), (150, 30), (0, 0, 0), -1)
        img = cv2.addWeighted(rectangle, 0.6, img, 0.4, 0)
        cv2.putText(img, "Action:", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if hidden_action == -1:
            color = (55, 155, 255)
            text = "LEFT"
        elif hidden_action == 1:
            text = "RIGHT"
            color = (55, 155, 255)
        elif hidden_action == 0:
            text = "STRAIGHT"
            color = (55, 255, 55)
        else:
            text = "UNKNOWN"
            color = (55, 55, 255)
        cv2.putText(img, text, (80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imshow(f'{curdata} - inference', img)

        if opts.show_base_images == 'True' and opts.num_components > 1:
            for i in range(opts.num_components * 2):
                img = base_imgs[i][0].cpu().numpy()
                img = np.rollaxis(img, 0, 3)
                img = ((img+1)*127.5).astype(np.uint8)
                img = cv2.resize(img, resized_image_size, interpolation=cv2.INTER_NEAREST)
                img = img[...,::-1]
                cv2.imshow(f'{curdata} - {"un" if i > opts.num_components - 1 else ""}masked {(i % opts.num_components) + 1}', img)
            for i in range(opts.num_components):
                img = m[i][0].cpu().numpy()
                img = np.rollaxis(img, 0, 3)
                img = ((img+1)*127.5).astype(np.uint8)
                img = np.expand_dims(cv2.resize(img, resized_image_size, interpolation=cv2.INTER_NEAREST), -1)
                img = img[...,::-1]
                cv2.imshow(f'{curdata} - mask {i + 1}', img)

        if upsample is not None:
            upsampled_img = upsample.inference(np.rollaxis(prev_state[0].cpu().numpy(), 0, 3))
            if type(upsampled_img) is np.ndarray:
                cv2.imshow(f'{curdata} - upsampled', upsampled_img[0][...,::-1])
            else:
                cv2.imshow(f'{curdata} - upsampled', upsampled_img[0][0][...,::-1])
                if len(upsampled_img) > 1:
                    cv2.imshow(f'{curdata} - upsampled aux1', upsampled_img[1][0][...,::-1])
                if len(upsampled_img) > 2:
                    cv2.imshow(f'{curdata} - upsampled aux2', upsampled_img[2][0][...,::-1])




        cv2.waitKey(1)

        i += 1

        # Uncomment to wite to the video stream
        #v.write(img)

        wait = 1/opts.playback_fps - (time.time() - frame_start_time)
        if wait > 0:
            time.sleep(wait)


if __name__ == '__main__':
    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)
    inference(opts.gpu, opts)
