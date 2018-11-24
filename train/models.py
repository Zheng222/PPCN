import tensorflow as tf
from ops import *

def teacher(input_image):
    input = adjust_contrast(input_image)
    with tf.variable_scope("generator_t"):
        W0 = weight_variable([1, 7, 3, 16], name="W0")  # shape=[filter_height, filter_width, in_channel, out_channel], name
        b0 = bias_variable([16], name="b0")
        c0 = lrelu(conv2d(input, W0) + b0)

        W1 = weight_variable([7, 1, 16, 16], name="W1")
        b1 = bias_variable([16], name="b1")
        c1 = lrelu(conv2d(c0, W1) + b1)

        out_c1 = tf.reduce_mean(c1, axis=-1, keepdims=True)

        # residual 1
        W2 = weight_variable([3, 3, 16, 32], name="W2")
        b2 = bias_variable([32], name="b2")
        c2 = lrelu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 32, 16], name="W3")
        b3 = bias_variable([16], name="b3")
        c3 = _instance_norm(conv2d(c2, W3) + b3) + c1  # 16

        # residual 2 dialted residual block
        W4 = weight_variable([3, 3, 16, 32], name="W4")
        b4 = bias_variable([32], name="b4")
        c4 = lrelu(_instance_norm(atrous_conv2d(c3, W4, 2) + b4))

        W5 = weight_variable([3, 3, 32, 16], name="W5")
        b5 = bias_variable([16], name="b5")
        c5 = _instance_norm(atrous_conv2d(c4, W5, 2) + b5) + c3  # 16

        gather = tf.concat([c3, c5], axis=-1)  ## 16*2=32
        W6 = weight_variable([1, 1, 32, 16], name="W6")
        b6 = bias_variable([16], name="b6")
        c6 = lrelu(conv2d(gather, W6) + b6)

        out_c6 = tf.reduce_mean(c6, axis=-1, keepdims=True)

        # Final
        W7 = weight_variable([3, 3, 16, 3], name="W7")
        b7 = bias_variable([3], name="b7")
        enhanced = tf.nn.tanh(conv2d(c6, W7) + b7) * 0.58 + 0.5

    return enhanced, out_c1, out_c6

def student(input_image):
    input = adjust_contrast(input_image)
    with tf.variable_scope("generator_s"):
        W0 = weight_variable([1, 7, 3, 8], name="W0")
        b0 = bias_variable([8], name="b0")
        c0 = lrelu(conv2d(input, W0) + b0)

        W1 = weight_variable([7, 1, 8, 8], name="W1")
        b1 = bias_variable([8], name="b1")
        c1 = lrelu(conv2d(c0, W1) + b1)

        out_c1 = tf.reduce_mean(c1, axis=-1, keepdims=True)

        W2 = weight_variable([1, 7, 8, 16], name="W2")
        b2 = bias_variable([16], name="b2")
        c2 = lrelu(_instance_norm(atrous_conv2d(c1, W2, 2) + b2))

        W3 = weight_variable([7, 1, 16, 8], name="W3")
        b3 = bias_variable([8], name="b3")
        c3 = lrelu(_instance_norm(atrous_conv2d(c2, W3, 2) + b3)) + c1

        out_c3 = tf.reduce_mean(c3, axis=-1, keepdims=True)

        # Final
        W4 = weight_variable([3, 3, 8, 3], name="W4")
        b4 = bias_variable([3], name="b4")
        enhanced = tf.nn.tanh(conv2d(c3, W4) + b4) * 0.58 + 0.5

    return enhanced, out_c1, out_c3

def adversarial(image_):
    with tf.variable_scope("discriminator"):
        conv1 = _conv_layer(image_, 64, 9, 4, batch_nn=False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 256, 3, 2)
        conv6 = _conv_layer(conv5, 256, 3, 1)

        flat_size = 256 * 7 * 7
        conv6_flat = tf.reshape(conv6, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv6_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)

    return adv_out

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)  # outputs random values from a truncated normal distribution
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):  # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # an input tensor of shape [batch, in_height, in_width, in_channels],
# a filter/kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]

def atrous_conv2d(x, W, rate):
    return tf.nn.atrous_conv2d(x, W, rate=rate, padding='SAME')

def depthwise_conv2d(x, W, rate=None):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', rate=rate)

def leaky_relu(x, alpha = 0.2):  # for discriminator
    return tf.maximum(alpha * x, x)

def lrelu(x, alpha = 0.05):  # for generator
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]  # [N, H, W, C]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)  # calculate the mean and variance of x
    # for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth],
    # pass axes=[0, 1, 2]
    # for simple batch normalization pass axes=[0] (batch only)

    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]  # the shape of the input tensor

    # [filter_height, filter_width, in_channels, out_channels]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init

def adjust_contrast(x):
    return tf.image.adjust_contrast(x, 1.2)