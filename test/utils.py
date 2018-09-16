from __future__ import print_function
import tensorflow as tf
from scipy import misc
import numpy as np
import threading
import psutil
import time
import os
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

min_ram = 0
max_ram = 0
stop_thread = False


def load_dped_test():

    IMAGE_SIZE = 100 * 100 * 3

    test_directory_phone = "dped/patches/iphone/"
    test_directory_dslr = "dped/patches/canon/"

    NUM_TEST_IMAGES = len(os.listdir(test_directory_phone))

    test_phone = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))
    test_dslr = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))

    for i in range(0, NUM_TEST_IMAGES):

        I = np.asarray(misc.imread(test_directory_phone + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        test_phone[i, :] = I

        I = np.asarray(misc.imread(test_directory_dslr + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        test_dslr[i, :] = I

    return test_phone, test_dslr


def check_ram(process):

    global max_ram

    while not stop_thread:

        ram_current = process.memory_info().rss / 1048576.0
        if ram_current > max_ram:
            max_ram = ram_current

        time.sleep(0.1)


def compute_running_time(task, model_file, img_dir):

    global process
    global stop_thread
    NUM_VAL_IMAGES = 4

    config = tf.ConfigProto(device_count={'GPU': 0})
    process = psutil.Process(os.getpid())
    avg_time = 0
    max_consumed_RAM = 0

    with tf.Session(config=config) as sess:

        with tf.gfile.FastGFile(model_file, 'rb') as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

            x_ = sess.graph.get_tensor_by_name('input:0')
            out_ = sess.graph.get_tensor_by_name('output:0')

        for i in range(NUM_VAL_IMAGES):

            print("\rImage " + str(i + 1) + " / " + str(NUM_VAL_IMAGES), end='')

            if task == "superres":
                image = misc.imresize(misc.imread(img_dir + str(i) + ".png"), 0.25, interp="bicubic")
                image = misc.imresize(image, 4.0, interp="bicubic")
            else:
                image = misc.imread(img_dir + str(i) + ".png")

            image = np.reshape(image, [1, image.shape[0], image.shape[1], 3]) / 255

            min_ram = process.memory_info().rss / 1048576.0
            stop_thread = False

            ram_thread = threading.Thread(target=check_ram, args=[process])
            ram_thread.start()

            time_start = int(round(time.time() * 1000))

            output = sess.run(out_, feed_dict={x_: image})

            time_finish = int(round(time.time() * 1000))
            stop_thread = True

            if i > 1:
                avg_time += (time_finish - time_start) / (NUM_VAL_IMAGES - 2)

            if max_ram - min_ram > max_consumed_RAM:
                max_consumed_RAM = max_ram - min_ram

    sess.close()

    print("\r\r\r")
    return int(avg_time), int(max_consumed_RAM)

def rgb2y(images):
    r, g, b = tf.split(images, 3, axis=-1)
    scale = 1 / 255
    y = (65.481 * r + 128.553 * g + 24.966 * b + 16) * scale
    return y

def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims

def shave(im, border):
    border=[border, border]
    im=im[border[0]:-border[0], border[1]:-border[1], ...]
    return im

def compute_psnr(im1,im2):
    p=psnr(im1,im2)
    return p

def compute_ssim(im1,im2):
    s=ssim(im1,im2,K1=0.01,K2=0.03,gaussian_weights=True,sigma=1.5,use_sample_covariance=False, multichannel=False)
    return s

def quantize(img):
    return np.uint8(img.clip(0, 255))
