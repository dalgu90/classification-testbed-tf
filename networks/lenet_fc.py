#!/usr/bin/env python

from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils

HParams = namedtuple('HParams',
                    'batch_size, num_classes, weight_decay, momentum')

class LeNet(object):
    def __init__(self, hp, images, labels, global_step, name='lenet'):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
        self._global_step = global_step
        self._name = name
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_model(self):
        print('Building model')
        with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')):
            filters = [500, 300]  # (784)-500-300-(10)

            # Input reshaping
            x = tf.reshape(self._images, [self._hp.batch_size, -1])

            for i, f in enumerate(filters):
                print('fc_%d: %d nodes' % ((i+1), f))
                # fc_x
                x = self._relu(self._fc(x, filters[0], name='fc_%d'%(i+1)), name='relu_%d'%(i+1))
            # Logit
            print('logit: %d nodes' % (self._hp.num_classes))
            x = self._fc(x, self._hp.num_classes, name='logit')

            self._logits = x

            # Probs & preds & acc
            self.probs = tf.nn.softmax(x, name='probs')
            self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
            ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
            zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
            correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
            self.acc = tf.reduce_mean(correct, name='acc')
            tf.summary.scalar('accuracy', self.acc)

            # Loss & acc
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
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

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn", no_scale=False):
        x = utils._bn(x, self.is_train, self._global_step, name, no_scale=no_scale)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _dropout(self, x, keep_prob, name="dropout"):
        x = utils._dropout(x, keep_prob, name)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)



