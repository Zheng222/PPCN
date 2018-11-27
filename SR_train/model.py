import tensorflow as tf


def Generator(t_image, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):

        input_mean = tf.reduce_mean(t_image, 2, keepdims=True)
        input_mean = tf.reduce_mean(input_mean, 1, keepdims=True)
        tensor = t_image - input_mean

        n = tf.layers.conv2d(tensor, 32, (3, 3), (2, 2), padding='same', activation=lrelu, name='downsample/1')
        n = tf.layers.conv2d(n, 64, (3, 3), (2, 2), padding='same', activation=lrelu, name='downsample/2')
        n = tf.layers.conv2d(n, 32, (3, 3), (1, 1), padding='same', name='n64s1/c1')
        temp = n

        # B residual blocks
        for i in range(2):
            n = ResBlock(n, name='res1/%i' % i)
        n = tf.layers.conv2d(n, 32, (3, 3), (1, 1), padding='same', name='trans/1')
        n += temp
        temp2 = n

        for i in range(2):
            n = ResBlock(n, name='res2/%i' % i)
        n = tf.layers.conv2d(n, 32, (3, 3), (1, 1), padding='same', name='trans/2')
        n += temp2

        n = tf.layers.conv2d(n, 32, (3, 3), (1, 1), padding='same', name='n64s1/c/m')
        n += temp
        # B residual blacks end

        n = upsample(n)
        n = tf.clip_by_value(n + input_mean, 0.0, 255.0)

    return n


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


def ResBlock(x, channels=32, kernel_size=3, name=''):
    tmp = tf.layers.conv2d(x, channels * 2, kernel_size, padding='same', name=name + '/conv1')
    tmp = lrelu(tmp)
    tmp = tf.layers.conv2d(tmp, channels, kernel_size, padding='same', name=name + '/conv2')
    return x + tmp


def upsample(x, scale=4, features=32):
    x = tf.layers.conv2d(x, features, 3, padding='same')
    ps_features = 3 * (4 ** 2)
    x = tf.layers.conv2d(x, ps_features, 3, padding='same')
    x = PS(x, scale, color=True)
    return x
