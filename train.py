#!/usr/bin/env python

import os
from datetime import datetime
import time

import tensorflow as tf
import numpy as np

from data import cifar10_input, cifar100_input, mnist_input, cifar10, mnist
from networks import lenet_fc, lenet_5, lenet_fc_test

# Dataset Configuration
tf.app.flags.DEFINE_string('dataset', 'cifar-10', """Dataset type.""")
tf.app.flags.DEFINE_string('data_dir', './cifar-10-binary/cifar-10-batches-bin/', """Path to the dataset.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_string('network', 'lenet', """Network architecture""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "10.0,20.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
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
            # When using cifar10(not cifar10_input), make sure that batch_size is a divisor of 10000
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    # train_images, train_labels = cifar10_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
                    train_images, train_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    # test_images, test_labels = cifar10_input.inputs(True, FLAGS.data_dir, FLAGS.batch_size)
                    test_images, test_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)
        elif 'cifar-100'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = cifar100_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = cifar100_input.inputs(True, FLAGS.data_dir, FLAGS.batch_size)
        elif 'mnist'==FLAGS.dataset:
            # When using mnist(not mnist_input), make sure that batch_size is a divisor of 10000
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    # train_images, train_labels = mnist_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
                    train_images, train_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    # test_images, test_labels = mnist_input.inputs(True, FLAGS.data_dir, FLAGS.batch_size)
                    test_images, test_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)

        # Build model
        if 'lenet-fc'==FLAGS.network:
            network = lenet_fc
        elif 'lenet-5'==FLAGS.network:
            network = lenet_5

        # 1) Training Network
        hp = network.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum)
        network_train = network.LeNet(hp, train_images, train_labels, global_step, name='train')
        network_train.build_model()
        network_train.build_train_op()

        train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # 2) Test network(reuse_variables!)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            network_test = network.LeNet(hp, test_images, test_labels, global_step, name='test')
            network_test.build_model()

        # Learning rate decay
        lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size for s in lr_decay_steps])
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
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

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
            train_images_val, train_labels_val = sess.run([train_images, train_labels])
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
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()