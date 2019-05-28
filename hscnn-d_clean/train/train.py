"""
Instantiate a solver for training
"""

import tensorflow as tf
from solver import *


flags = tf.app.flags

# settings of the solver and hyper-parameters
flags.DEFINE_string('train_data', '/home/custom/train_ntire18_new.tfrecord', 'path of the training dataset')
flags.DEFINE_integer('num_train_exps', 148556, 'number examples per epoch for training')
flags.DEFINE_string('models_dir', './models', 'trained model save path')
flags.DEFINE_string('log_name', './models/spsr.log', 'name of the log file')

flags.DEFINE_string('valid_data', '/home/custom/strain_ntire18_new.tfrecord', 'path of the validation dataset')
flags.DEFINE_integer('num_valid_exps', 7020, 'number examples per epoch for validation')

# setting of the gpu devices
flags.DEFINE_integer('device_id', 0, 'assign the first id number')
flags.DEFINE_integer('num_gpus', 8, 'how many GPUs to use')

# policy of the leanrning rate
flags.DEFINE_float('base_lr', 0.001, 'the basic (initial) learning rate')
flags.DEFINE_float('power', 1.5, 'power of the polynomial')
flags.DEFINE_float('end_lr', 0.00001, 'the minimal end learning rate')

# several multiplier
flags.DEFINE_float('loss_weight', 100.0, 'the multiplier of loss')
flags.DEFINE_float('lr_mp', 1.0, 'the multiplier of learning rate')
flags.DEFINE_float('decay_fraction', 1.0, 'the factor of decay policy')

# setting of warming-up
flags.DEFINE_integer('warmup_epoch', 0, 'number of epochs for warming up learning rate')
flags.DEFINE_boolean('warmup_from0', False, 'whether to warm up from zero')

# epoch and batch size
flags.DEFINE_integer('num_epoch', 60, 'number of training epoch')
flags.DEFINE_integer('batch_size', 64, 'batch size of the training dataset')
flags.DEFINE_integer('valid_bs', 64, 'batch size of the valid dataset')

# regularization
flags.DEFINE_float('weight_decay', None, 'weight decay')
flags.DEFINE_boolean('non_local', False, 'whether to enable the non-local block')

# resuming and finetune
flags.DEFINE_boolean('resume', False, 'whether to resume from the trained variables')
flags.DEFINE_boolean('finetune', False, 'whether to finetune from the trained variables')
flags.DEFINE_string('finetune_models_dir', './models', 'the path for searching the trained model')
flags.DEFINE_integer('iters', None, 'iteration of the trained variable')
flags.DEFINE_boolean('meta_data', False, 'whether to log the meta-data (need to load the libcupti.so)')

conf = flags.FLAGS


def main(_):
	solver = Solver()
	solver.train(disp_freq=100, save_freq=200, valid_freq=200)


if __name__ == '__main__':
	tf.app.run()
