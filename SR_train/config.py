from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 2e-4

## Generator
config.TRAIN.n_epoch = 5000
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 1000

## training dataset location
config.TRAIN.hr_img_path = '/data/DIV2K_train_HR/'
config.TRAIN.checkpoint_dir = 'checkpoint/'

config.VALID = edict()
config.VALID.hr_img_path = '/data/DIV2K_valid_HR/'
config.VALID.lr_img_path = '/data/DIV2K_valid_LR_x4/'
