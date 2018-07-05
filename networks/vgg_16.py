#!/usr/bin/env python

from collections import namedtuple

import tensorflow as tf
import numpy as np

from .network import Network
import utils

HParams = namedtuple('HParams',
                    'batch_size, num_classes, fc_bias, weight_decay, momentum')
HParams.__new__.__defaults__ = (100, 10, True, 0.0001, 0.9)

class LeNet(Network):
    def __init__(self, hp, images, labels, global_step, name='lenet'):
        super(LeNet, self).__init__(hp, images, labels, global_step, name)

    def build_model(self):
        print('Building model')
        with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')):
            x = self._images
            if len(x.get_shape()) == 2:
                from functools import reduce
                from operator import mul
                data_len = reduce(mul, x.get_shape()[1:].as_list())
                if data_len == 784:
                    x = tf.reshape(x, [self._hp.batch_size, 28, 28, 1])
                elif data_len == 3072:
                    x = tf.reshape(x, [self._hp.batch_size, 32, 32, 3])
                else:
                    print('Input cannot be reshaped into 3-dim')
                    return

            def conv_bn_relu(x, filter_size, out_channel, name_suffix):
                x = self._conv(x, filter_size, out_channel, 1, pad="SAME", trainable=True, name="conv"+name_suffix)
                x = self._bn(x, no_scale=False, trainable=True, name="bn"+name_suffix)
                x = self._relu(x, name="relu"+name_suffix)
                return x

            x = conv_bn_relu(x, 3, 64, name_suffix="1")
            x = self._dropout(x, 0.7, name="dropout1")
            x = conv_bn_relu(x, 3, 64, name_suffix="2")
            x = self._max_pool(x, 2, 2, name="pool2")

            x = conv_bn_relu(x, 3, 128, name_suffix="3")
            x = self._dropout(x, 0.6, name="dropout3")
            x = conv_bn_relu(x, 3, 128, name_suffix="4")
            x = self._max_pool(x, 2, 2, name="pool4")

            x = conv_bn_relu(x, 3, 256, name_suffix="5")
            x = self._dropout(x, 0.6, name="dropout5")
            x = conv_bn_relu(x, 3, 256, name_suffix="6")
            x = self._dropout(x, 0.6, name="dropout6")
            x = conv_bn_relu(x, 3, 256, name_suffix="7")
            x = self._max_pool(x, 2, 2, name="pool7")

            x = conv_bn_relu(x, 3, 512, name_suffix="8")
            x = self._dropout(x, 0.6, name="dropout8")
            x = conv_bn_relu(x, 3, 512, name_suffix="9")
            x = self._dropout(x, 0.6, name="dropout9")
            x = conv_bn_relu(x, 3, 512, name_suffix="10")
            x = self._max_pool(x, 2, 2, name="pool10")

            x = conv_bn_relu(x, 3, 512, name_suffix="11")
            x = self._dropout(x, 0.6, name="dropout11")
            x = conv_bn_relu(x, 3, 512, name_suffix="12")
            x = self._dropout(x, 0.6, name="dropout12")
            x = conv_bn_relu(x, 3, 512, name_suffix="13")
            x = self._max_pool(x, 2, 2, name="pool13")

            x = tf.reshape(x, [self._hp.batch_size, -1])
            x = self._dropout(x, 0.5, name="dropout13")
            x = self._fc(x, 512, bias=self._hp.fc_bias, name='fc14')
            x = self._bn(x, name="bn14")
            x = self._relu(x, name='relu14')
            x = self._dropout(x, 0.5, name="dropout15")
            x = self._fc(x, self._hp.num_classes, bias=self._hp.fc_bias, name='fc15')

            self._logits = x

            # Probs & preds & acc
            self.probs = tf.nn.softmax(self._logits, name='probs')
            self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
            ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
            zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
            correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
            self.acc = tf.reduce_mean(correct, name='acc')
            tf.summary.scalar('accuracy', self.acc)

            # Loss & acc
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('cross_entropy', self.loss)


    def build_train_op(self):
        print('Build training ops')

        with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')):
            # Learning rate
            tf.summary.scalar('learing_rate', self.lr)

            losses = [self.loss]

            # Add l2 loss
            with tf.variable_scope('l2_loss'):
                costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                losses.append(l2_loss)

            self._total_loss = tf.add_n(losses)

            # Gradient descent step
            opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
            grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
            # if self._hp.finetune:
            # for idx, (grad, var) in enumerate(grads_and_vars):
                # if "unit3" in var.op.name or \
                # "unit_last" in var.op.name or \
                # "logits" in var.op.name:
                # print('Scale up learning rate of % s by 10.0' % var.op.name)
                # grad = 10.0 * grad
            # grads_and_vars[idx] = (grad,var)
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                with tf.control_dependencies(update_ops+[apply_grad_op]):
                    self.train_op = tf.no_op()
            else:
                self.train_op = apply_grad_op



