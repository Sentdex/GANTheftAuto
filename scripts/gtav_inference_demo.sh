#!/usr/bin/env bash

python3 inference.py \
 --saved_model ./trained_models/gan_5_1_17.pt \
 --data gtav:./data/gtav/gtagan_2_sample \
 --inference_image_path ./data/gtav/2.png \
 --show_base_images True \
 --upsample_model ./trained_models/upsample---[_20]---[______3171]---[_____63420].h5 \