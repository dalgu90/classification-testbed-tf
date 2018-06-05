#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./lenet-5_cifar100"

python train.py --train_dir $train_dir \
    --network "lenet-5" \
    --dataset "cifar-100" \
    --data_dir "data/cifar-100-binary/" \
    --num_classes 100 \
    --num_train_instance 50000 \
    --num_test_instance 10000 \
    --batch_size 500 \
    --test_interval 100 \
    --test_iter 20 \
    --l2_weight 0.0005 \
    --initial_lr 0.05 \
    --lr_step_epoch 100.0,200.0 \
    --lr_decay 0.1 \
    --max_steps 30000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 10 \
