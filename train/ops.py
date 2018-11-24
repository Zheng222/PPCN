import tensorflow as tf
import tensorflow.contrib as tf_contirb

weight_init = tf_contirb.layers.variance_scaling_initializer() # Kaiming init
weight_regularizer = tf_contirb.layers.l2_regularizer(scale=1e-4)

## Layer

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if scope.__contains__('discriminator'):
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else:
            weight_init = tf_contirb.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)
        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def flatten(x):
    return tf.layers.flatten(x)

## Residual Block

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x_gate = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, scope='gate')
            x_gate = instance_norm(x_gate, scope='instance_norm_gate')
            x_gate = softmax(x_gate)

            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = tf.multiply(x_gate, relu(x))

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

    return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, scope='adaptive_resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)

    return x + x_init
## Sampling

def down_sample(x):
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')


## Activation function

def lrelu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.tanh(x)

def softmax(x):
    return tf.nn.softmax(x)

## Normalization function
def adaptive_instance_norm(content, gamma, beta, epsilon=1e-3):
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def instance_norm(x, scope='instance_norm'):
    return tf_contirb.layers.instance_norm(x,
                                           epsilon=1e-3,
                                           center=True, scale=True,
                                           scope=scope)