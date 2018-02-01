import numpy as np
import tensorflow as tf
import time
import pudb
import argparse
from tensorflow.python.client import timeline


# Arguments
parser = argparse.ArgumentParser(description='Slice Test')
parser.add_argument('--type', type=str, metavar='M', default='normal',
                    help='Types of layers(normal, slice)')
parser.add_argument('--layers', type=int, default=10,
                    help='Number of fc layers in the network')
parser.add_argument('--nodes', type=int, default=4096,
                    help='Number of nodes in each fc layer')
parser.add_argument('--group', type=int, default=2,
                    help='Number of groups in each fc layer(only for slice type)')
parser.add_argument('--split_inner', type=str, default="False",
                    help='Number of groups in each fc layer(only for slice type)')
parser.add_argument('--even_slice', type=str, default="False",
                    help='True to slice layers evenly, false to slice at random positions')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--iter', type=int, default=500,
                    help='Number of iterations to test')
parser.add_argument('--tracing', type=bool, default=False,
                    help='Whether to output timeline')
args = parser.parse_args()

# Parse arguments
N_layers = args.layers
N = args.batch_size
D = args.nodes
G = args.group
N_iter = args.iter
Tracing = args.tracing
Split_inner = True if args.split_inner=="True" else False
Even_slice = True if args.even_slice=="True" else False


def _fc(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/x.get_shape().as_list()[1])))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc

def _fc_group(x, in_dims, out_dims, slice_input=True, concat_output=True, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        w = tf.get_variable('weights', [D, D], tf.float32,
                            initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/D)))
        b = tf.get_variable('biases', [D], tf.float32,
                            initializer=tf.constant_initializer(0.0))

        if slice_input:
            temp_idx = 0
            x_list = []
            for i in range(G):
                x_list.append(tf.slice(x, [0, temp_idx], [N, in_dims[i]], name='x_slice_%d'%(i+1)))
            temp_idx += in_dims[i]
        else:
            x_list = x

        fc_list = []
        temp_idx_in = temp_idx_out = 0
        for i in range(G):
            w_slice = tf.slice(w, [temp_idx_in, temp_idx_out], [in_dims[i], out_dims[i]], name='w_slice_%d'%(i+1))
            b_slice = tf.slice(b, [temp_idx_out], [out_dims[i]], name='b_slice_%d'%(i+1))
            fc_slice = tf.nn.bias_add(tf.matmul(x_list[i], w_slice), b_slice)
            fc_list.append(fc_slice)
            temp_idx_in += in_dims[i]
            temp_idx_out += out_dims[i]

        if concat_output:
            fc = tf.concat(fc_list, axis=1)
        else:
            fc = fc_list

    return fc


def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def _relu_group(xs, leakness=0.0, name=None):
    if type(xs) is not list:
        return _relu(xs, leakness, name)

    relu_list = []
    with tf.variable_scope(name):
        for i, x in enumerate(xs):
            relu_list.append(_relu(x, leakness, name='relu_%d'%(i+1)))
    return relu_list


def simple_net(x):
    h = x
    for i in range(N_layers-1):
        h = _relu(_fc(h, D, name='fc_%d'%(i+1)), name='relu_%d'%(i+1))
    pred = _fc(h, D, name='fc_pred')
    return pred

# def slice_net(x):
    # dist = tf.distributions.Categorical(probs=np.ones(D-G+1)/(D-G+1))
    # def get_dims():
        # if not Even_slice:
            # ps = range(1,G) + tf.contrib.framework.sort(dist.sample(G-1))
            # dims = tf.concat([ps, tf.constant([D], tf.int32)], axis=0) - tf.concat([tf.constant([0], tf.int32), ps], axis=0)
        # else:
            # dims = [(i+1)*D/G-i*D/G for i in range(G)]
        # return dims

    # in_dims = get_dims()
    # out_dims = get_dims()
    # h = _relu_group(_fc_group(x, in_dims, out_dims, True, Split_inner, name='fc_1'), name='relu_1')
    # for i in range(N_layers-2):
        # in_dims = out_dims
        # out_dims = get_dims()
        # h = _relu_group(_fc_group(h, in_dims, out_dims, Split_inner, Split_inner, name='fc_%d'%(i+2)), name='relu_%d'%(i+2))
    # in_dims = out_dims
    # out_dims = get_dims()
    # pred = _fc_group(h, in_dims, out_dims, Split_inner, True, name='fc_pred')

    # return pred

def slice_net(x):
    if not Even_slice:
        dist = tf.distributions.Categorical(probs=np.ones(D-G+1)/(D-G+1))
        ps = range(1, G) + tf.contrib.framework.sort(dist.sample((N_layers+1, G-1)))
        dims = tf.concat([ps, tf.constant(D*np.ones((N_layers+1,1), np.int32), tf.int32)], axis=1) \
            - tf.concat([tf.constant(np.zeros((N_layers+1,1), np.int32), tf.int32), ps], axis=1)
    else:
        dims = [[(i+1)*D/G-i*D/G for i in range(G)] for _ in range(N_layers+1)]

    h = _relu_group(_fc_group(x, dims[0], dims[1], True, not Split_inner, name='fc_1'), name='relu_1')
    for i in range(1, N_layers-1):
        h = _relu_group(_fc_group(h, dims[i], dims[i+1], not Split_inner, not Split_inner, name='fc_%d'%(i+1)), name='relu_%d'%(i+1))
    pred = _fc_group(h, dims[N_layers-1], dims[N_layers], not Split_inner, True, name='fc_pred')

    return pred

def main():
    print('%d layers' % N_layers)
    print('%d nodes' % D)
    print('%s network' % (args.type))
    if args.type == 'slice':
        print('groups: %d' % G)
        print('split inner: %s' % str(Split_inner))
        print('even slice: %s' % str(Even_slice))
    print('batch size: %d' % N)
    print('%d iterations' % N_iter)

    # Build graph
    x = tf.placeholder(tf.float32, (N, D))
    y = tf.placeholder(tf.float32, (N, D))

    if args.type == 'normal':
        pred = simple_net(x)
    elif args.type == 'slice':
        pred = slice_net(x)
    loss = tf.nn.l2_loss(y - pred)

    # Feed_dict
    x_val = np.random.rand(N, D)
    y_val = np.random.rand(N, D)
    feed_dict = {x:x_val, y:y_val}

    # Create session
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    start_time = time.time()

    for i in range(N_iter):
        if i % 100 == 99:
            print(i+1)

        if i != 50 or not Tracing:
            loss_val = sess.run(loss, feed_dict=feed_dict)
        else:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            loss_val = sess.run(loss, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

    end_time = time.time()

    print('Elapsed time: %.6fs' % (end_time - start_time))

if __name__=='__main__':
    main()
