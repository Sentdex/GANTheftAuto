"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import os
import sys
import numpy as np
import torch.utils.data as data_utils
import cv2
import random
import pickle
import gzip


sys.path.insert(0, './data')
sys.path.append('../../')
sys.path.append('../')
import utils

def get_custom_dataset(opts=None, set_type=0, force_noshuffle=False, getLoader=True, num_workers=1):

    shuffle = True if set_type == 0 else False
    shuffle = True if opts.play else shuffle

    if force_noshuffle:
        shuffle = False

    curdata, datadir = opts.data.split(':')
    if 'pacman' == curdata or 'pacman_maze' == curdata:
        dataset = berkeley_pacman_dataset(opts, set_type=set_type, datadir=datadir)
    elif 'vizdoom' == curdata:
        dataset = vizdoom_dataset(opts,set_type=set_type, datadir=datadir)
    elif 'cartpole' == curdata:
        dataset = cartpole_dataset(opts, set_type=set_type, datadir=datadir)
    elif 'vroom' == curdata:
        dataset = vroom_dataset(opts, set_type=set_type, datadir=datadir)
    elif 'gtav' == curdata:
        dataset = gtav_dataset(opts, set_type=set_type, datadir=datadir)
    else:
        print('Unsupported Dataset')
        exit(-1)

    if getLoader:
        dloader = data_utils.DataLoader(dataset, batch_size=opts.bs,
                num_workers=num_workers, pin_memory=False, shuffle=shuffle, drop_last=True)
        return dloader
    else:
        return dataset

class vizdoom_dataset(data_utils.Dataset):

    def __init__(self, opts, set_type=0,  permute_color=False, datadir=''):
        self.opts = opts
        self.set_type = set_type
        self.permute_color = permute_color

        self.samples = []
        num_data = len(os.listdir(datadir))
        if set_type == 0:
            sample_list = list(range(0, int(num_data*0.9)))
        else:
            sample_list = list(range(int(num_data*0.9), num_data))
        for el in sample_list:
            self.samples.append('%s/%d.npy' % (datadir, el))
        self.end_bias = 0
        if utils.check_arg(self.opts, 'end_bias'):
            self.end_bias = self.opts.end_bias

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn = self.samples[idx]
        data = np.load(fn, allow_pickle=True).item()
        states, actions, neg_actions = [], [], []

        ep_len = len(data['obs']) - self.opts.num_steps

        if random.random() < self.end_bias:
            start_pt = random.randint(ep_len - max(2, ep_len//50), ep_len - 1)
        else:
            start_pt = random.randint(0, ep_len - 1)

        i = 0
        while i < self.opts.num_steps:
            try:
                if start_pt + i >= len(data['obs']):
                    cur_s = data['obs'][len(data['obs']) - 1]
                    cur_a = data['action'][len(data['obs']) - 1]
                else:
                    cur_s = data['obs'][start_pt + i]
                    cur_a = data['action'][start_pt + i]
            except:
                import pdb;
                pdb.set_trace()

            action = [0] * self.opts.action_space
            if cur_a > 0.33333:
                action[0] = 1
            elif cur_a < -0.33333:
                action[1] = 1
            else:
                action[2] = 1

            s_t = (np.transpose(cur_s, axes=(2, 0, 1)) / 255.).astype('float32')
            s_t = (s_t - 0.5) / 0.5
            a_t = np.asarray(action).astype('float32')

            action_idx = a_t.tolist().index(1)

            # sample false action
            false_a_idx = random.randint(0, 2)
            while false_a_idx == action_idx:
                false_a_idx = random.randint(0, 2)
            false_a_t = np.zeros(a_t.shape).astype('float32')
            false_a_t[false_a_idx] = 1

            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            i = i + 1

        return states, actions, neg_actions


class berkeley_pacman_dataset(data_utils.Dataset):

    def __init__(self, opts, set_type=0, permute_color=False, datadir=''):
        self.opts = opts
        self.set_type = set_type
        self.permute_color = permute_color

        self.samples = []
        num_data = len(os.listdir(datadir))
        if set_type == 0:
            sample_list = list(range(0, int(num_data * 0.9)))
        else:
            sample_list = list(range(int(num_data * 0.9), num_data))

        for el in sample_list:
            self.samples.append('%s/%d.npy' % (datadir, el))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], encoding='latin1')
        states, actions, neg_actions = [], [], []
        ep_len = len(data[0]['np_img_state']) - self.opts.num_steps

        start_pt = random.randint(0, max(ep_len - 1, 0))

        i = 0
        samples = []
        cur_sample = 0
        while i < self.opts.num_steps:
            try:
                if start_pt + i >= len(data[0]['np_img_state'])-2:
                    cur_s = data[0]['np_img_state'][len(data[0]['np_img_state']) - 2]
                    cur_a = np.zeros(self.opts.action_space)
                    cur_a[0] = 1
                else:
                    cur_s = data[0]['np_img_state'][start_pt + i]
                    cur_a = data[0]['np_action'][start_pt + i]
            except:
                import pdb; pdb.set_trace()


            if self.opts.img_size[0] != cur_s.shape[1] or self.opts.img_size[1] != cur_s.shape[0]:
                cur_s = cv2.resize(cur_s.astype('float32'),
                            dsize=(self.opts.img_size[0], self.opts.img_size[1]),)

            s_t = (np.transpose(cur_s, axes=(2, 0, 1)) / 255.).astype('float32')

            s_t = (s_t - 0.5) / 0.5
            if utils.check_arg(self.opts, 'normalize_mean'):
                s_t = s_t - np.array([-0.9219, -0.9101, -0.8536]).reshape((3,1,1)).astype('float32')
                s_t = s_t/1.9219

            a_t = np.copy(cur_a).astype('float32')
            action_idx = cur_a.tolist().index(1)

            # false action
            false_a_idx = random.randint(0, 4)
            while false_a_idx == action_idx:
                false_a_idx = random.randint(0, 4)
            false_a_t = np.zeros(cur_a.shape).astype('float32')
            false_a_t[false_a_idx] = 1

            # save
            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            samples.append(cur_sample)
            i = i + 1


        return states, actions, neg_actions

def cp_vizdoom_data(data_dir, out_dir):
    import shutil
    i = 0
    for num in os.listdir(data_dir):
        cur_dir = os.path.join(data_dir, num)
        if not os.path.isdir(cur_dir):
            continue
        for cur_f in os.listdir(cur_dir):
            shutil.copyfile(os.path.join(cur_dir, cur_f), os.path.join(out_dir, str(i)+'.npy'))
            i += 1

# Custom, cartpole data loader
class cartpole_dataset(data_utils.Dataset):

    # Initialization, almost teh same as for other data types
    def __init__(self, opts, set_type=0, permute_color=False, datadir=''):

        self.opts = opts
        self.set_type = set_type
        self.permute_color = permute_color

        self.samples = []
        files = os.listdir(datadir)
        num_data = len(files)
        if set_type == 0:
            sample_list = files[:int(num_data * 0.9)]
        else:
            sample_list = files[int(num_data * 0.9):]

        # Here's the difference - we're using gzipped pickle
        # (additionally checking if teh file exists)
        for file in sample_list:
            path = f'{datadir}/{file}'
            self.samples.append(path)

        # Bias
        self.end_bias = self.opts.end_bias if utils.check_arg(self.opts, 'end_bias') else 0.5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # Load the sequence
        with gzip.open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # Initialize data lists and calculate the episode length
        states, actions, neg_actions = [], [], []
        ep_len = len(data['observations']) - self.opts.num_steps

        # Using bias, decide if to draw a random subsequence from the whole dataset
        # or from the end (when probably model fails)
        if ep_len <= 1:
            start_pt = 0
        elif random.random() < self.end_bias:
            start_pt = random.randint(max(0, ep_len - self.opts.num_steps*2), ep_len - 1)
        else:
            start_pt = random.randint(0, ep_len - 1)

        i = 0
        samples = []
        cur_sample = 0

        # Iterate over num_steps steps
        while i < self.opts.num_steps:

            # If current index exceedes number of steps - use last step
            if start_pt + i >= len(data['observations']):
                cur_s = data['observations'][len(data['observations']) - 1]
                cur_a = data['actions'][len(data['observations']) - 1]
            # Or given step otherwise
            else:
                cur_s = data['observations'][start_pt + i]
                cur_a = data['actions'][start_pt + i]

            # Channels last -> channels first, scale to -1..1, save as float32
            s_t = (np.transpose(cur_s, axes=(2, 0, 1)) / 255.).astype('float32')
            s_t = (s_t - 0.5) / 0.5

            # Code sparse label as one-hot vector
            a_t = np.eye(2)[cur_a].astype('float32')

            # Since we have just 2 actions, false action is always the other one
            false_a_t = np.eye(2)[1-cur_a].astype('float32')

            # Add to the lists
            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            samples.append(cur_sample)
            i = i + 1

        # Return data
        return states, actions, neg_actions

# Custom, vroom data loader
class vroom_dataset(data_utils.Dataset):

    # Initialization, almost teh same as for other data types
    def __init__(self, opts, set_type=0, permute_color=False, datadir=''):

        self.opts = opts
        self.set_type = set_type
        self.permute_color = permute_color

        self.samples = []
        files = os.listdir(datadir)
        num_data = len(files)
        if set_type == 0:
            sample_list = files[:int(num_data * 0.9)]
        else:
            sample_list = files[int(num_data * 0.9):]

        # Here's the difference - we're using gzipped pickle
        # (additionally checking if teh file exists)
        for file in sample_list:
            path = f'{datadir}/{file}'
            self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # Load the sequence
        with gzip.open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # Initialize data lists and calculate the episode length
        states, actions, neg_actions = [], [], []
        ep_len = len(data['observations']) - self.opts.num_steps

        # Find sub-sequence start point
        start_pt = random.randint(0, ep_len - 1)

        i = 0
        samples = []
        cur_sample = 0

        # Iterate over num_steps steps
        while i < self.opts.num_steps:

            # If current index exceedes number of steps - use last step
            if start_pt + i >= len(data['observations']):
                cur_s = data['observations'][len(data['observations']) - 1]
                cur_a = data['actions'][len(data['observations']) - 1]
            # Or given step otherwise
            else:
                cur_s = data['observations'][start_pt + i]
                cur_a = data['actions'][start_pt + i]

            # Channels last -> channels first, scale to -1..1, save as float32
            s_t = (np.transpose(cur_s, axes=(2, 0, 1)) / 255.).astype('float32')
            s_t = (s_t - 0.5) / 0.5

            # Code sparse label as one-hot vector
            a_t = np.eye(3)[cur_a].astype('float32')
            action_idx = cur_a

            # false action
            false_a_idx = random.randint(0, 2)
            while false_a_idx == action_idx:
                false_a_idx = random.randint(0, 2)
            false_a_t = np.zeros(a_t.shape).astype('float32')
            false_a_t[false_a_idx] = 1

            # Add to the lists
            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            samples.append(cur_sample)
            i = i + 1

        # Return data
        return states, actions, neg_actions

# Custom, vroom data loader
class gtav_dataset(data_utils.Dataset):

    # Initialization, almost teh same as for other data types
    def __init__(self, opts, set_type=0, permute_color=False, datadir=''):

        self.opts = opts
        self.set_type = set_type
        self.permute_color = permute_color

        self.samples = []
        files = os.listdir(datadir)
        num_data = len(files)
        if set_type == 0:
            sample_list = files[:int(num_data * 0.9)]
        else:
            sample_list = files[int(num_data * 0.9):]

        # Here's the difference - we're using gzipped pickle
        # (additionally checking if teh file exists)
        for file in sample_list:
            path = f'{datadir}/{file}'
            self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # Load the sequence
        with gzip.open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # Initialize data lists and calculate the episode length
        states, actions, neg_actions = [], [], []
        ep_len = len(data['observations']) - self.opts.num_steps

        # Find sub-sequence start point
        start_pt = random.randint(0, ep_len - 1)

        i = 0
        samples = []
        cur_sample = 0

        # Iterate over num_steps steps
        while i < self.opts.num_steps:

            # If current index exceedes number of steps - use last step
            if start_pt + i >= len(data['observations']):
                cur_s = data['observations'][len(data['observations']) - 1]
                cur_a = data['actions'][len(data['observations']) - 1]
            # Or given step otherwise
            else:
                cur_s = data['observations'][start_pt + i]
                cur_a = data['actions'][start_pt + i]

            # Channels last -> channels first, scale to -1..1, save as float32
            s_t = (np.transpose(cur_s, axes=(2, 0, 1)) / 255.).astype('float32')
            s_t = (s_t - 0.5) / 0.5

            # Code sparse label as one-hot vector
            a_t = np.eye(3)[cur_a].astype('float32')
            action_idx = cur_a

            # false action
            false_a_idx = random.randint(0, 2)
            while false_a_idx == action_idx:
                false_a_idx = random.randint(0, 2)
            false_a_t = np.zeros(a_t.shape).astype('float32')
            false_a_t[false_a_idx] = 1

            # Add to the lists
            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            samples.append(cur_sample)
            i = i + 1

        # Return data
        return states, actions, neg_actions

if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cp_vizdoom_data(data_dir, out_dir)
