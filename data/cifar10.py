import tensorflow as tf
import os

NUM_CLASSES = 10
INIT_LRN_RATE = 1e-2
MIN_LRN_RATE = 1e-4
WEIGHT_DECAY_RATE = 1e-4
RELU_LEAKINESS = 0.1
NUM_TRAIN_IMAGES = 50000

HEIGHT = 32
WIDTH = 32
DEPTH = 3

NEW_HEIGHT = 32
NEW_WIDTH = 32

cifar10_mean = [125.3, 123.0, 113.9]
cifar10_std = [63.0, 62.1, 66.7]


def LRN_scheduler(hParams, FLAGS, epoch):
    if (epoch < 0.5*FLAGS.n_epochs):
        lrn_rate = hParams.lrn_rate
    elif (epoch < 0.8*FLAGS.n_epochs):
        lrn_rate = hParams.lrn_rate * 1e-1
    elif (epoch < 1.0*FLAGS.n_epochs):
        lrn_rate = hParams.lrn_rate * 1e-2
    else:
        lrn_rate = hParams.lrn_rate * 1e-3
    return lrn_rate

def record_dataset(filenames):
    """Returns an input pipeline Dataset from `filenames`."""
    record_bytes = HEIGHT * WIDTH * DEPTH + 1
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)

def get_filenames(data_dir, train_mode):
    """Returns a list of filenames based on 'mode'."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    if train_mode:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, 6)
        ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]

def dataset_parser(value):
    label_bytes = 1
    image_bytes = HEIGHT * WIDTH * DEPTH
    record_bytes = label_bytes + image_bytes

    raw_record = tf.decode_raw(value, tf.uint8)
    label = tf.cast(raw_record[0], tf.int32)

    depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
                           [DEPTH, HEIGHT, WIDTH])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    return image, label
    # return image, tf.one_hot(label, NUM_CLASSES)

def train_preprocess_fn(image, label):
    image = tf.image.resize_image_with_crop_or_pad(image, NEW_HEIGHT+4, NEW_WIDTH+4)
    image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, 3])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - cifar10_mean) / cifar10_std
    return image, label

def test_preprocess_fn(image, label):
    # image = tf.image.resize_images(image, [NEW_HEIGHT+4, NEW_WIDTH+4])
    # image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, 3])
    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - cifar10_mean) / cifar10_std
    return image, label

def input_fn(dataset, batch_size, train_mode, num_threads=8):
    dataset = record_dataset(get_filenames(dataset, train_mode))
    dataset = dataset.repeat()
    dataset = dataset.map(dataset_parser, num_parallel_calls=num_threads)

    if train_mode:
        buffer_size = int(50000 * 0.4) + 3 * batch_size
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(train_preprocess_fn, batch_size))
    else:
        dataset = dataset.repeat()
        dataset = dataset.apply(tf.contrib.data.map_and_batch(test_preprocess_fn, batch_size))

    dataset = dataset.prefetch(3)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    images.set_shape((batch_size, NEW_WIDTH, NEW_HEIGHT, DEPTH))
    labels.set_shape((batch_size, ))

    return images, labels
