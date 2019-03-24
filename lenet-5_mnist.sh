#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./results/lenet-5_mnist"

python train.py --train_dir $train_dir \
    --network "lenet-5" \
    --dataset "mnist-aug" \
    --data_dir "data/mnist/" \
    --num_train_instance 60000 \
    --num_classes 10 \
    --batch_size 100 \
    --test_interval 600 \
    --test_iter 100 \
    --fc_bias False \
    --l2_weight 0.0001 \
    --initial_lr 0.05 \
    --lr_step_epoch 100.0 \
    --lr_decay 0.1 \
    --max_steps 120000 \
    --checkpoint_interval 12000 \
    --gpu_fraction 0.95 \
    --display 100 \
