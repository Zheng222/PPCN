import tensorflow as tf
import tensorlayer as tl
from model import Generator
from config import config
import numpy as np
import os
from tensorboardX import SummaryWriter
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
checkpoint_dir = config.TRAIN.checkpoint_dir
hr_image_path = config.TRAIN.hr_img_path
scale = 4
alpha_ssim = 25
eval_every = 10

with tf.variable_scope(tf.get_variable_scope()):
    ## create folders to save trained model
    tl.files.exists_or_mkdir(checkpoint_dir)

    ## pre-load data
    train_hr_npy = os.path.join(hr_image_path, 'train_hr.npy')
    valid_hr_npy = os.path.join(config.VALID.hr_img_path, 'valid_hr.npy')
    valid_lr_npy = os.path.join(config.VALID.lr_img_path, 'valid_lr_x{}.npy'.format(scale))

    if os.path.exists(train_hr_npy) and os.path.exists(valid_hr_npy) and os.path.exists(valid_lr_npy):
        print('Loading data...')
        train_hr_imgs = np.load(train_hr_npy)
        valid_hr_imgs = np.load(valid_hr_npy)
        valid_lr_imgs = np.load(valid_lr_npy)
    else:
        print('Creating data binary...')
        train_hr_imgs_list = sorted(tl.files.load_file_list(path=hr_image_path, regx='.*.png', printable=False))
        valid_hr_imgs_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))

        train_hr_imgs = np.array(tl.visualize.read_images(train_hr_imgs_list, path=hr_image_path, n_threads=16))
        valid_hr_imgs = np.array(tl.visualize.read_images(valid_hr_imgs_list, path=config.VALID.hr_img_path, n_threads=16))
        valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=utils.downsample_fn, scale=scale)

        np.save(train_hr_npy, train_hr_imgs)
        np.save(valid_hr_npy, valid_hr_imgs)
        np.save(valid_lr_npy, valid_lr_imgs)

    ## define model
    tensor_lr = tf.placeholder('float32', [None, None, None, 3], name='tensor_lr')
    tensor_hr = tf.placeholder('float32', [None, None, None, 3], name='tensor_hr')

    print('Loading model...')
    tensor_sr = Generator(tensor_lr)

    ## calculate the number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters: %d" % total_parameters)


    ## define loss functions
    mae_loss = tf.reduce_mean(tf.losses.absolute_difference(tensor_sr, tensor_hr))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(tensor_sr, tensor_hr, max_val=255))

    g_loss = mae_loss + alpha_ssim * ssim_loss

    ## PSNR and SSIM (Evaluation)
    PSNR = tf.image.psnr(tensor_sr, tensor_hr, max_val=255)
    SSIM = tf.image.ssim_multiscale(tensor_sr, tensor_hr, max_val=255)

## create the optimization
g_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]

with tf.variable_scope("learning_rate"):
    lr_value = tf.Variable(lr_init, trainable=False)
g_optim = tf.train.AdamOptimizer(learning_rate=lr_value).minimize(g_loss, var_list=g_vars)

## restore model
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# load from checkpoint if exist
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

## Tensorboard
writer = SummaryWriter(os.path.join(checkpoint_dir, 'result'))
tf.summary.FileWriter(os.path.join(checkpoint_dir, 'graph'), sess.graph)
best_psnr, best_epoch = 0, 0

## training
print("Training network...")
for epoch in range(0, n_epoch + 1):
    # update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        new_lr_decay = lr_decay ** (epoch // decay_every)
        sess.run(tf.assign(lr_value, lr_init * new_lr_decay))
        log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
        print(log)
    elif epoch == 0:
        sess.run(tf.assign(lr_value, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
        print(log)

    index = np.random.permutation(len(train_hr_imgs))
    num_batches = len(train_hr_imgs) // batch_size

    total_losses = np.zeros(3)
    for i in range(num_batches):
        hr = tl.prepro.threading_data(train_hr_imgs[index[i * batch_size: (i+1) * batch_size]], fn=utils.crop_sub_imgs_fn, is_random=True)
        lr = tl.prepro.threading_data(hr, fn=utils.downsample_fn, scale=scale)
        [lr, hr] = utils.datatype([lr, hr])
        ## update G
        error_g, error_mae, error_ssim, _ = sess.run([g_loss, mae_loss, ssim_loss, g_optim], {tensor_lr: lr, tensor_hr: hr})
        total_losses += [error_g, error_mae, error_ssim]

    avg_loss = total_losses / num_batches
    log = "[*] Epoch: [%2d/%2d] g_loss: %.6f, mae: %.6f, ssim: %.6f" % \
          (epoch, n_epoch, avg_loss[0], avg_loss[1], avg_loss[2])
    print(log)

    writer.add_scalar('g_loss', avg_loss[0], epoch)
    writer.add_scalar('mae_loss', avg_loss[1], epoch)
    writer.add_scalar('ssim_loss', avg_loss[2], epoch)

    ## validating
    if (epoch != 0 and epoch % eval_every == 0):
        print('Validating...')
        val_psnr = 0
        val_ssim = 0
        for i in range(len(valid_hr_imgs)):
            hr = valid_hr_imgs[i]
            lr = valid_lr_imgs[i]
            [lr, hr] = utils.datatype([lr, hr])

            hr_expand = np.expand_dims(hr, axis=0)
            lr_expand = np.expand_dims(lr, axis=0)

            psnr, ssim, sr_expand = sess.run([PSNR, SSIM, tensor_sr], {tensor_lr: lr_expand, tensor_hr: hr_expand})
            sr = np.squeeze(sr_expand)
            utils.update_tensorboard(epoch, writer, i, lr, sr, hr)
            val_psnr += psnr
            val_ssim += ssim

        val_psnr = val_psnr / len(valid_hr_imgs)
        val_ssim = val_ssim / len(valid_hr_imgs)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            print('Saving new best model')

            ## save model
            saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        writer.add_scalar('Validate PSNR', val_psnr, epoch)
        writer.add_scalar('Validate SSIM', val_ssim, epoch)
