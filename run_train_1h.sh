#!/bin/bash
echo "========================================"
echo "Training Transformer với IWSLT trong 1 giờ"
echo "========================================"
python train_fast.py --max_time_hours 1.0 --batch_size 128 --d_model 256 --n_heads 4 --n_encoder_layers 3 --n_decoder_layers 3 --d_ff 1024 --max_len 64 --use_amp

