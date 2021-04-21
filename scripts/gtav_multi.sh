python #!/usr/bin/env bash

python3 --data gtav:./data/gtav/gtagan_1 \
 --log_dir ./results/ \
 --num_steps 32 \
 --warm_up 16 \
 --warmup_decay_epoch 60 \
 --bs 1 \
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
 --img_size 80x48 \
 #--simple_blocks \
 --action_space 3 #\
 #--saved_model ./results/model6.pt \
 #--saved_optim ./results/optim176.pt