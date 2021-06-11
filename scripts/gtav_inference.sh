#!/usr/bin/env bash

python3 inference.py \
 --saved_model ./results/model3.pt \
 --data gtav:./data/gtav/data \
 --inference_image_path ./data/gtav/2.png \
 --show_base_images True \