#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./lenet-mnist-scratch"

python train.py --train_dir $train_dir \
    --network "lenet-fc" \
    --dataset "mnist" \
    --data_dir "data/mnist/" \
    --num_train_instance 60000 \
    --num_classes 10 \
    --batch_size 500 \
    --test_interval 120 \
    --test_iter 20 \
    --l2_weight 0.0005 \
    --initial_lr 0.1 \
    --lr_step_epoch 50.0,100.0 \
    --lr_decay 0.1 \
    --max_steps 18000 \
    --checkpoint_interval 3000 \
    --gpu_fraction 0.96 \
    --display 10 \
