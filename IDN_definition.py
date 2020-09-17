# Code from https://github.com/Zheng222/IDN-tensorflow/blob/master/model.py
import tensorflow as tf

def IDN(t_image, t_image_bicubic, scale, reuse=False):
    t_image_bicubic = tf.identity(t_image_bicubic)
    with tf.variable_scope("IDN", reuse=reuse):
        conv1 = tf.layers.conv2d(t_image, 64, (3, 3), (1, 1), padding='same', activation=lrelu, name='conv1')
        conv2 = tf.layers.conv2d(conv1, 64, (3, 3), (1, 1), padding='same', activation=lrelu, name='conv2')
        n = conv2
        for i in range(4):
            n = distillation(n, name='distill/%i' % i)
        output = upsample(n, scale=scale,features=64, name=str(scale)) + t_image_bicubic
    return output

def distillation(x, name=''):
    tmp = tf.layers.conv2d(x, 48, (3, 3), (1, 1), padding='same', activation=lrelu, name=name+'/conv1')
    tmp = GroupConv2d(tmp, act=lrelu, name=name+'/conv2')
    tmp = tf.layers.conv2d(tmp, 64, (3, 3), (1, 1), padding='same', activation=lrelu, name=name+'/conv3')
    tmp1, tmp2 = tf.split(axis=3, num_or_size_splits=[16, 48], value=tmp)
    tmp2 = tf.layers.conv2d(tmp2, 64, (3, 3), (1, 1), padding='same', activation=lrelu, name=name+'/conv4')
    tmp2 = GroupConv2d(tmp2, n_filter=48, act=lrelu, name=name+'/conv5')
    tmp2 = tf.layers.conv2d(tmp2, 80, (3, 3), (1, 1), padding='same', activation=lrelu, name=name+'/conv6')
    output = tf.concat(axis=3, values=[x, tmp1]) + tmp2
    output = tf.layers.conv2d(output, 64, (1, 1), (1, 1), padding='same', activation=lrelu, name=name+'/conv7')
    return output


def lrelu(x, alpha=0.05):
    return tf.maximum(alpha * x, x)


def _phase_shift(I, r):
    return tf.depth_to_space(I, r)


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)  # tf.split(value, num_or_size_splits, axis=0)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X

def upsample(x, scale=4, features=32, name=None):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, features, 3, padding='same')
        ps_features = 3 * (scale ** 2)
        x = tf.layers.conv2d(x, ps_features, 3, padding='same')
        x = PS(x, scale, color=True)
    return x

def GroupConv2d(x, n_filter=32, filter_size=(3, 3), strides=(1, 1), n_group=4, act=None, padding='SAME', name=None):
    groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strides[0], strides[1], 1], padding=padding)
    channels = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        We = tf.get_variable(
            name='W', shape=[filter_size[0], filter_size[1], channels / n_group, n_filter], trainable=True
        )

        if n_group == 1:
            outputs = groupConv(x, We)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=n_group, value=x)
            weightsGroups = tf.split(axis=3, num_or_size_splits=n_group, value=We)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

            outputs = tf.concat(axis=3, values=convGroups)

        b = tf.get_variable(
            name='b', shape=n_filter, trainable=True
        )

        outputs = tf.nn.bias_add(outputs, b, name='bias_add')

        if act:
            outputs = lrelu(outputs)
    return outputs