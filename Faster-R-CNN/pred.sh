#!/bin/bash
python validation.py \
--device cuda \
--num-classes 4 \
--data-path /home/lab/Python_pro/Tianchi/Dataset \
--weights ./save_weights/resNetFpn-model-18.pth \
--batch_size 4