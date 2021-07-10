#!/bin/bash

python train.py \
--data-path /home/lab/Python_pro/Tianchi/Dataset \
--batch_size 4 \
--num-classes 4 \
--epochs 20 \
--output-dir ./save_weights
