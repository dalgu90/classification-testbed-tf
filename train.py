#!/usr/bin/env python

import os
from datetime import datetime
import time
from six.moves import xrange

import tensorflow as tf
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data
from data import cifar10, cifar100, mnist
from networks import lenet_fc, lenet_5, vgg_16

# Dataset Configuration
tf.app.flags.DEFINE_string('dataset', 'mnist', """Dataset type.""")
tf.app.flags.DEFINE_string('data_dir', './data/mnist/', """Path to the dataset.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 60000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_string('network', 'lenet-fc', """Network architecture""")
tf.app.flags.DEFINE_boolean('fc_bias', True, """Whether to add bias after fc multiply""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "100.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 120000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 600, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.96, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tDataset: %s' % FLAGS.dataset)
    print('\tDataset dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tNetwork architecture: %s' % FLAGS.network)
    print('\tFC layer bias: %d' % FLAGS.fc_bias)
    print('\tBatch size: %d' % FLAGS.batch_size)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per testing: %d' % FLAGS.test_interval)
    print('\tSteps during testing: %d' % FLAGS.test_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels
        if 'cifar-10'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)
        elif 'cifar-100'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = cifar100.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = cifar100.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)
        elif 'mnist'==FLAGS.dataset:
            # Tensorflow default dataset
            mnist_read = input_data.read_data_sets(FLAGS.data_dir, one_hot=False, validation_size=0)
            def train_func():
                return mnist_read.train.next_batch(FLAGS.batch_size, shuffle=True)
            def test_func():
                return mnist_read.test.next_batch(FLAGS.batch_size, shuffle=False)
            train_images, train_labels = tf.py_func(train_func, [], [tf.float32, tf.uint8])
            train_images.set_shape([FLAGS.batch_size, 784])
            train_labels.set_shape([FLAGS.batch_size])
            train_labels = tf.cast(train_labels, tf.int32)
            test_images, test_labels = tf.py_func(test_func, [], [tf.float32, tf.uint8])
            test_images.set_shape([FLAGS.batch_size, 784])
            test_labels.set_shape([FLAGS.batch_size])
            test_labels = tf.cast(test_labels, tf.int32)
        elif 'mnist-aug'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)


        # Build model
        if 'lenet-fc'==FLAGS.network:
            network = lenet_fc
        elif 'lenet-5'==FLAGS.network:
            network = lenet_5
        elif 'vgg-16'==FLAGS.network:
            network = vgg_16

        # 1) Training Network
        hp = network.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            fc_bias=FLAGS.fc_bias,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum)
        network_train = network.LeNet(hp, train_images, train_labels, global_step, name='train')
        network_train.build_model()
        network_train.build_train_op()

        print('FLOPs: %d' % network_train._flops)
        print('Weigths: %d' % network_train._weights)

        train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # 2) Test network(reuse_variables!)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            network_test = network.LeNet(hp, test_images, test_labels, global_step, name='test')
            network_test.build_model()

        # Learning rate decay
        lr_decay_steps = [float(s) for s in FLAGS.lr_step_epoch.split(',')]
        lr_decay_steps = [int(f) for f in [s*FLAGS.num_train_instance/FLAGS.batch_size for s in lr_decay_steps]]
        def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
            lr = initial_lr
            for s in lr_decay_steps:
                if global_step >= s:
                    lr *= lr_decay
            return lr

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)
            print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))))

        # Training!
        test_best_acc = 0.0
        for step in xrange(init_step, FLAGS.max_steps):
            # Test
            if step % FLAGS.test_interval == 0:
                test_loss, test_acc = 0.0, 0.0
                for i in range(FLAGS.test_iter):
                    loss_value, acc_value = sess.run([network_test.loss, network_test.acc],
                                                     feed_dict={network_test.is_train:False})
                    test_loss += loss_value
                    test_acc += acc_value
                test_loss /= FLAGS.test_iter
                test_acc /= FLAGS.test_iter
                test_best_acc = max(test_best_acc, test_acc)
                format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, test_loss, test_acc))

                test_summary = tf.Summary()
                test_summary.value.add(tag='test/loss', simple_value=test_loss)
                test_summary.value.add(tag='test/acc', simple_value=test_acc)
                test_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
                summary_writer.add_summary(test_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
            start_time = time.time()
            _, lr_value, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.lr, network_train.loss, network_train.acc, train_summary_op],
                        feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
