import tensorflow as tf
import os

NUM_CLASSES = 10
INIT_LRN_RATE = 1e-2
MIN_LRN_RATE = 1e-4
WEIGHT_DECAY_RATE = 1e-4
RELU_LEAKINESS = 0.1
NUM_TRAIN_IMAGES = 50000
NUM_TEST_IMAGES = 10000

HEIGHT = 28
WIDTH = 28
DEPTH = 1

NEW_HEIGHT = 28
NEW_WIDTH = 28

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
    # return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    return tf.contrib.data.FixedLengthRecordDataset(filenames, record_bytes)

def get_filenames(data_dir, train_mode):
    """Returns a list of filenames based on 'mode'."""
    data_dir = os.path.join(data_dir)

    if train_mode:
        return [os.path.join(data_dir, 'train.bin')]
    else:
        return [os.path.join(data_dir, 'test.bin')]

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
    image = tf.image.resize_images(image, [NEW_HEIGHT+4, NEW_WIDTH+4])
    image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, DEPTH])
    image = tf.image.random_flip_left_right(image)
    return image, label

def test_preprocess_fn(image, label):
    image = tf.image.resize_images(image, [NEW_HEIGHT+4, NEW_WIDTH+4])
    image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, DEPTH])
    return image, label

def input_fn(dataset, batch_size, train_mode, num_threads=4):
    dataset = record_dataset(get_filenames(dataset, train_mode))
    dataset = dataset.repeat()

    dataset = dataset.map(dataset_parser, num_threads=num_threads,
                        output_buffer_size=3*batch_size)

    if train_mode:
        dataset = dataset.map(train_preprocess_fn, num_threads=num_threads,
                              output_buffer_size=3*batch_size)
    else:
        dataset = dataset.map(test_preprocess_fn, num_threads=num_threads,
                                output_buffer_size=3*batch_size)
    buffer_size = int(50000 * 0.4) + 3 * batch_size
    if train_mode:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(
      lambda image, label: (tf.image.per_image_standardization(image), label),
      num_threads=num_threads,
      output_buffer_size=3*batch_size)

    # iterator = dataset.batch(batch_size).make_one_shot_iterator()
    # Make sure that the number of data is divisible by batch_size
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels
