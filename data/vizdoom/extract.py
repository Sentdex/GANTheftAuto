"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
Contains some code from:
https://github.com/hardmaru/WorldModelsExperiments/tree/master/doomrnn
with the following license:

The MIT License (MIT)

Copyright (c) hardmaru

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



import numpy as np
import random
import os
import gym
from doomreal import _process_frame
from env import make_env
from model import make_model
from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'activation'])
doomreal = Game(env_name='doomreal',
  input_size=576+512*1,
  output_size=1,
  activation='tanh',
)


def extract(opts):
  MAX_FRAMES = opts.max_length  # from doomtakecover
  MAX_TRIALS = opts.max_trials  # just use this to extract one trial.
  MIN_LENGTH = opts.min_length

  render_mode = False # for debugging.

  DIR_NAME = opts.save_dir
  if not os.path.exists(DIR_NAME):
      os.makedirs(DIR_NAME)

  model = make_model(doomreal)

  fixed_repeat = opts.fixed_repeat
  if fixed_repeat > 1:
    MIN_LENGTH = MIN_LENGTH // fixed_repeat
  total_frames = 0

  use_predicted_action = True if opts.cma_model_path != '' else False
  model.make_env(render_mode=render_mode, load_model=False)  # random weights

  if opts.visdom:
    import vis_util
    vis_win = None
    vis_util.visdom_initialize(4212, 'local')

  num_saved = 0
  for trial in range(MAX_TRIALS):
    try:
      random_generated_int = random.randint(0, 2**31-1)
      recording_obs = []
      recording_action = []

      np.random.seed(random_generated_int)
      model.env.seed(random_generated_int)

      # random policy
      # model.init_random_model_params(stdev=0.2)
      # more diverse random policy, works slightly better:

      if not use_predicted_action:
        if fixed_repeat > 1:
          multiplier = np.random.randint(1, 12 // fixed_repeat + 1)
          repeat = fixed_repeat * multiplier
        else:
          repeat = np.random.randint(1, 11)

      obs = model.env.reset()
      pixel_obs = model.env.current_obs # secret 64x64 obs frame


      save = False
      for frame in range(MAX_FRAMES):
        if render_mode:
          model.env.render("human")
        if not use_predicted_action:
          if frame % repeat == 0:
            action = np.random.rand() * 2.0 - 1.0
            if fixed_repeat > 1:
              multiplier = np.random.randint(1, 12//fixed_repeat + 1)
              repeat = fixed_repeat * multiplier
            else:
              repeat = np.random.randint(1, 11)

          if fixed_repeat is not None:
            if frame % fixed_repeat == 0:
              recording_obs.append(pixel_obs)
              recording_action.append(action)
          else:
            recording_obs.append(pixel_obs)
            recording_action.append(action)
        else:
          action = model.get_action(obs)
          if frame % fixed_repeat == 0:
            recording_obs.append(pixel_obs)
            recording_action.append(action)
            saved = True
          else:
            saved = False
        obs, reward, done, save = model.env._step(action, extract=True)

        pixel_obs = model.env.current_obs # secret 64x64 obs frame

        if done:
          break

      total_frames += frame
      print("dead at", frame, "total recorded frames for this worker", total_frames)
      recording_obs = np.array(recording_obs, dtype=np.uint8)
      recording_action = np.array(recording_action, dtype=np.float16)
      if (len(recording_obs) > MIN_LENGTH):
        filename = DIR_NAME + "/" + str(num_saved) + ".npy"
        newdata = {'obs': recording_obs, 'action': recording_action}
        np.save(filename, newdata)
        num_saved += 1
        if num_saved == opts.num_generate:
          break
    except gym.error.Error:
      print("stupid doom error, life goes on")
      model.env.close()
      model.make_env(render_mode=render_mode)
      continue
  model.env.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  parser.add_argument('--min_length', type=int, default=300, help='initial seed')
  parser.add_argument('--max_length', type=int, default=700, help='initial seed')
  parser.add_argument('--max_trials', type=int, default=13000000, help='initial seed')
  parser.add_argument('--num_generate', type=int, default=11000, help='initial seed')
  parser.add_argument('--fixed_repeat', type=int, default=3, help='initial seed')
  parser.add_argument(
    '--env_model_path', default='', metavar='LG', help='folder to save logs')
  parser.add_argument(
    '--cma_model_path', default='', metavar='LG', help='folder to save logs')
  parser.add_argument(
    '--envname', default='doomreal', metavar='LG', help='folder to save logs')
  parser.add_argument(
    '--save_dir', default='tmp', metavar='LG', help='folder to save logs')
  parser.add_argument(
    '--visdom',
    default=False,
    metavar='R',
    help='Watch game as it being played')
  opts = parser.parse_args()

  extract(opts)
