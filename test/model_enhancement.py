import tensorflow as tf

def student_generator(input_image):
    input = adjust_contrast(input_image)
    with tf.variable_scope("generator_s"):
        W0 = weight_variable([1, 7, 3, 16], name="W0")  # shape=[filter_height, filter_width, in_channel, out_channel], name
        b0 = bias_variable([16], name="b0")
        c0 = lrelu(conv2d(input, W0) + b0)

        W1 = weight_variable([7, 1, 16, 16], name="W1")
        b1 = bias_variable([16], name="b1")
        c1 = lrelu(conv2d(c0, W1) + b1)

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
        # Convolutional
        W6 = weight_variable([1, 1, 32, 16], name="W6")
        b6 = bias_variable([16], name="b6")
        c6 = lrelu(conv2d(gather, W6) + b6)

        # Final
        W7 = weight_variable([3, 3, 16, 3], name="W7")
        b7 = bias_variable([3], name="b7")
        enhanced = tf.nn.tanh(conv2d(c6, W7) + b7) * 0.58 + 0.5  # [-0.08, 0.18]

    return enhanced

def student_generator(input_image):
    input = adjust_contrast(input_image)
    with tf.variable_scope("generator_little"):
        W0 = weight_variable([1, 7, 3, 8], name="W0")
        b0 = bias_variable([8], name="b0")
        c0 = lrelu(conv2d(input, W0) + b0)

        W1 = weight_variable([7, 1, 8, 8], name="W1")
        b1 = bias_variable([8], name="b1")
        c1 = lrelu(conv2d(c0, W1) + b1)

        W2 = weight_variable([1, 7, 8, 16], name="W2")
        b2 = bias_variable([16], name="b2")
        c2 = lrelu(_instance_norm(atrous_conv2d(c1, W2, 2) + b2))

        W3 = weight_variable([7, 1, 16, 8], name="W3")
        b3 = bias_variable([8], name="b3")
        c3 = lrelu(_instance_norm(atrous_conv2d(c2, W3, 2) + b3)) + c1

        # Final
        W4 = weight_variable([3, 3, 8, 3], name="W4")
        b4 = bias_variable([3], name="b4")
        enhanced = tf.nn.tanh(conv2d(c3, W4) + b4) * 0.58 + 0.5

    return enhanced


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)  # outputs random values from a truncated normal distribution
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')


def atrous_conv2d(x, W, rate):
    return tf.nn.atrous_conv2d(x, W, rate=rate, padding='SAME')


def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]  # [N, H, W, C]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)  # calculate the mean and variance of x

    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)

    return scale * normalized + shift


def lrelu(x, alpha=0.05):
    return tf.maximum(alpha * x, x)


def adjust_contrast(x):
    return tf.image.adjust_contrast(x, 1.2)
