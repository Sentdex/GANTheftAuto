#!/usr/bin/env bash

python main_parallel.py \
 --data vizdoom:./data/vizdoom \
 --log_dir ./results/ \
 --num_steps 32 \
 --warm_up 16 \
 --warmup_decay_epoch 100 \
 --bs 6 \
 --num_components 1 \
 --fixed_v_dim 8 \
 --att_dim 8 \
 --fine_mask \
 --config_temporal 24 \
 --save_epoch 20 \
 --num_gpu 4 \
 --nfilterG 32 \
 --seed 111111 \
 --img_size 64 \
 --end_bias 0.5 \
