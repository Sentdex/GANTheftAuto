#!/usr/bin/env bash

python3 main_parallel.py \
 --data vroom:./data/vroom/data \
 --log_dir ./results/ \
 --num_steps 32 \
 --warm_up 16 \
 --warmup_decay_epoch 60 \
 --bs 2 \
 --num_components 2 \
 --fine_mask \
 --config_temporal 32 \
 --do_memory \
 --cycle_loss \
 --alpha_loss_multiplier 0.000075 \
 --softmax_kernel \
 --sigmoid_maps \
 --save_epoch 1 \
 --rev_multiply_map \
 --num_gpu 1 \
 --temperature 0.1 \
 --nfilterG 16 \
 --spade_index 2 \
 --seed 1 \
 --img_size 64 \
 --simple_blocks \
 --action_space 3
