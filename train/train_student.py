import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import models
import utils
import vgg
import os
from CX.CX_helper import *
from CX.enums import Distance
from easydict import EasyDict as edict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, lr_init, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

decay_every = 10000
lr_decay = 0.1
w_kd = 75

config_CX = edict()
config_CX.crop_quarters = False
config_CX.max_sampling_1d_size = 65
config_CX.Dist = Distance.DotProduct
config_CX.nn_stretch_sigma = 0.5
config_CX.batch_size = batch_size

np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced, at1, at2 = models.student(phone_image)
    enhanced_teacher, at1_, at2_ = models.teacher(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image), [-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)  # if adv_ = 0, enhanced_gray
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)  # if adv_=0, [0, 1], train generator

    loss_discrim = -tf.reduce_sum(discrim_target * tf.log(
        tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))  # Note: here use tf.reduce_sum, not use tf.reduce_mean
    loss_texture = -loss_discrim

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # 2) content loss

    CX_LAYER = 'conv4_2'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    # SSIM loss
    ssim_loss = 25 * (1 - utils.ssim(dslr_image, enhanced) / batch_size)

    # CX loss
    cx_loss = 4 * CX_loss_helper(dslr_vgg[CX_LAYER], enhanced_vgg[CX_LAYER], config_CX)

    # content loss
    loss_content = ssim_loss + cx_loss

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2)) / (2 * batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:, 1:, :, :])  # H
    tv_x_size = utils._tensor_size(enhanced[:, :, 1:, :])  # W
    y_tv = tf.nn.l2_loss(enhanced[:, 1:, :, :] - enhanced[:, :batch_shape[1] - 1, :, :])
    x_tv = tf.nn.l2_loss(enhanced[:, :, 1:, :] - enhanced[:, :, :batch_shape[2] - 1, :])
    loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

    # 5) knowledge distillation loss
    stu = [at1, at2]
    tea = [at1_, at2_]
    loss_kd = utils.kd_loss(stu, tea) / batch_size

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv + w_kd * loss_kd

    # PSNR evaluation

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2)) / (PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.global_variables() if
                      v.name.startswith("generator_s")]
    discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    train_step_gen = tf.train.AdamOptimizer(learning_rate=lr_v).minimize(loss_generator, var_list=generator_vars)
    train_step_disc = tf.train.AdamOptimizer(learning_rate=lr_v).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    generator_t_vars = [v for v in tf.global_variables() if
                        v.name.startswith("generator_t")]
    saver_teacher = tf.train.Saver(var_list=generator_t_vars, max_to_keep=1)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    # load teacher net from checkpoint
    ckpt = tf.train.get_checkpoint_state('pretrained_teacher_model/')

    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver_teacher.restore(sess, ckpt.model_checkpoint_path)

    print('Training network')

    ckpt = tf.train.get_checkpoint_state('models_student/')
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_loss_gen = 0.0
    train_acc_discrim = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models_student/' + phone + '.txt', "w+")
    logs.close()

    for i in range(num_train_iters):

        # train generator

        ## update learning rate
        if i != 0 and (i % decay_every == 0):
            new_lr_decay = lr_decay ** (i // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f " % (lr_init * new_lr_decay)
            print(log)
        elif i == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f decay_every_init: %d, lr_decay: %f " % (lr_init, decay_every, lr_decay)
            print(log)

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                     feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
        train_loss_gen += loss_temp / eval_step

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_discrim, train_step_disc],
                                     feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
        train_acc_discrim += loss_temp / eval_step

        if i % eval_step == 0:

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 7))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0
            teacher_ssim = 0.0

            for j in range(num_test_batches):
                be = j * batch_size
                en = (j + 1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, enhanced_teacher_crops, accuracy_disc, losses] = sess.run(
                    [enhanced, enhanced_teacher, discim_accuracy,
                     [loss_generator, loss_content, loss_kd, loss_color, loss_texture, loss_tv, loss_psnr]],
                    feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})

                test_losses_gen += np.asarray(losses) / num_test_batches
                test_accuracy_disc += accuracy_disc / num_test_batches

                loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                            enhanced_crops * 255) / num_test_batches

                teacher_ssim += MultiScaleSSIM(
                    np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                    enhanced_teacher_crops * 255) / num_test_batches

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                        (i, phone, train_acc_discrim, test_accuracy_disc)

            logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, kd: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g" % \
                       (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                        test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], test_losses_gen[0][6],
                        loss_ssim)

            logs_teacher = "teacher | ssim: %.4g\n" % teacher_ssim

            print(logs_disc)
            print(logs_gen)
            print(logs_teacher)

            # save the results to log file

            logs = open('models_student/' + phone + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(phone) + "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models_student/' + str(phone) + '_iteration_' + str(i) + '.ckpt',
                       write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
