from __future__ import print_function
import tensorflow as tf
from scipy import misc
import numpy as np
import utils
import os
import skimage.io as sio

## --------- Change test parameters below -----------

from model_super_resolution import Generator as test_model              # import your model definition as "test_model"
model_location = "ckpt_super_resolution/generator_all.ckpt"    # specify the location of your saved pre-trained model (ckpt file)

## configurations
compute_PSNR_SSIM = True
compute_running_time = False
test_data_dir = 'SR_test_data/'
dataset = 'originla/'  # original(several DIV2K valid images) | Set5 | Set14 | B100 | Urban100 | manga109
output_folder = 'results/' + dataset
RGB = True # if True, compute psnr and ssim on RGB channels, else only on y channel

if __name__ == "__main__":

    print("\n-------------------------------------\n")
    print("Image Super-resolution task\n")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.warnings.filterwarnings('ignore')

    ###############################################################
    #  1 Produce .pb model file that will be used for validation  #
    ###############################################################

    print("Saving pre-trained model as .pb file")

    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:

        image_ = tf.placeholder(tf.float32, shape=(1, None, None, 3), name="input")
        out_ = tf.identity(test_model(image_), name="output")

        saver = tf.train.Saver()
        saver.restore(sess, model_location)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, g.as_graph_def(), "input,output".split(",")
        )

        tf.train.write_graph(output_graph_def, 'models_converted', 'model.pb', as_text=False)

    print("Model was successfully saved!")
    print("\n-------------------------------------\n")
    sess.close()


    if compute_PSNR_SSIM:

        #######################################
        #  2 Computing PSNR / MS-SSIM scores  #
        #######################################

        tf.reset_default_graph()
        config = None

        with tf.Session(config=config) as sess:

            print("\rLoading pre-trained model")

            with tf.gfile.FastGFile("models_converted/model.pb", 'rb') as f:

                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

                x_ = sess.graph.get_tensor_by_name('input:0')
                out_ = sess.graph.get_tensor_by_name('output:0')

            y_ = tf.placeholder(tf.float32, [1, None, None, 3])
            h_ = tf.placeholder(tf.int32)
            w_ = tf.placeholder(tf.int32)

            # Remove boundaries (4px) from the produced and target images

            if RGB:
                output_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(out_, 4, 4, h_, w_), 0.0, 1.0)
                target_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(y_, 4, 4, h_, w_), 0.0, 1.0)
            else:
                output_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(utils.rgb2y(out_), 4, 4, h_, w_), 0.0, 1.0)
                target_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(utils.rgb2y(y_), 4, 4, h_, w_), 0.0, 1.0)

            psnr_ = tf.image.psnr(output_crop_, target_crop_, max_val=1.0)  # psnr
            ssim_ = tf.image.ssim(output_crop_, target_crop_, max_val=1.0)  # ssim

            print("Computing PSNR/SSIM scores....")

            ssim_score = 0.0
            psnr_score = 0.0
            validation_images = os.listdir(test_data_dir + dataset)
            num_val_images = len(validation_images)

            for j in range(num_val_images):

                print("\rImage %d / %d" % (j + 1, num_val_images), end='')
                image = misc.imread(test_data_dir + dataset + validation_images[j])
                image = utils.modcrop(image, modulo=4)
                if len(image.shape) < 3:
                    image = image[..., np.newaxis]
                    image = np.concatenate([image] * 3, 2)

                image_bicubic = misc.imresize(image, 0.25, interp="bicubic")
                image_bicubic = misc.imresize(image_bicubic, 4.0, interp="bicubic")

                image_bicubic = np.reshape(image_bicubic, [1, image_bicubic.shape[0], image_bicubic.shape[1], 3]) / 255
                image_target = np.reshape(image, [1, image.shape[0], image.shape[1], 3]) / 255

                h = image.shape[0] - 8
                w = image.shape[1] - 8

                [psnr, ssim, output_image] = sess.run([psnr_, ssim_, out_], feed_dict={x_: image_bicubic, y_: image_target, h_: h, w_: w})

                psnr_score += psnr / num_val_images
                ssim_score += ssim / num_val_images

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                sio.imsave(output_folder + validation_images[j], utils.quantize(np.squeeze(output_image, axis=0) * 255.0))


            print("\r\r\r")
            print("Scores | PSNR: %.4g, MS-SSIM: %.4g" % (psnr_score, ssim_score))
            print("\n-------------------------------------\n")
            sess.close()


    if compute_running_time:

        ##############################
        #  3 Computing running time  #
        ##############################

        print("Evaluating model speed")
        print("This can take a few minutes\n")

        tf.reset_default_graph()

        print("Testing pre-trained baseline SRCNN model")
        avg_time_baseline, max_ram = utils.compute_running_time("superres", "models_pretrained/div2k_srcnn.pb", test_data_dir + "HD_res/")

        tf.reset_default_graph()

        print("Testing provided model")
        avg_time_solution, max_ram = utils.compute_running_time("superres", "models_converted/model.pb", test_data_dir + "HD_res/")

        print("Baseline SRCNN time, ms: ", avg_time_baseline)
        print("Test model time, ms: ", avg_time_solution)
        print("Speedup ratio (baseline, ms / solution, ms): %.4f" % (float(avg_time_baseline) / avg_time_solution))
        print("Approximate RAM consumption (HD image): " + str(max_ram) + " MB")
