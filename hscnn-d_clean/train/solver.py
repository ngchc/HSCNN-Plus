"""
The solver for training
"""

import tensorflow as tf
import numpy as np

import logging
from time import time
from os import path, makedirs

from data_loader import *
from model import Net


flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
	def _tower_loss(self, hr, lr, tower_index, reuse_variables, non_local):
		"""Calculate the total loss on a single tower running the model (with the batch splitting)
		Args:
		  datas:  4D tensor of size [batch_size, 1, image_size, image_size]
		  labels: 1-D integer Tensor of [batch_size]
		  scope: unique prefix string identifying the tower, e.g. 'tower_0'
		
		Returns:
		  tensor of shape [] containing the total loss for a batch of data
		"""
		# build the inference graph
		with tf.variable_scope(tf.get_variable_scope()):
			net = Net(hr, lr, non_local, wl=self.weight_decay, tower=tower_index, reuse=reuse_variables)
			net.build_net()
		
		# return the total loss for the current tower
		return net.total_loss
	
	
	def _average_gradients(self, tower_grads):
		"""calculate the average gradient for each shared variable across all towers
		Args:
		  tower_grads: list of lists of (gradient, variable) tuples
					   the outer list is over individual gradients
					   the inner list is over the gradient calculation for each tower

		Returns:
		  list of pairs of (gradient, variable) 
		  where the gradient has been averaged across all towers
		"""
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			# note: each grad_and_vars looks like the following:
			# ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
			grads = []
			for g, _ in grad_and_vars:
				# add 0 dimension to the gradients to represent the tower
				expanded_g = tf.expand_dims(g, 0)

				# append on a 'tower' dimension which we will average over below
				grads.append(expanded_g)

			# average over the 'tower' dimension
			grad = tf.concat(axis=0, values=grads)
			grad = tf.reduce_mean(grad, 0)

			# the variables are redundant, so just return the first tower's pointer to the variable
			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)

		return average_grads
	
	
	def _get_data(self, mode):
		if mode == 'train':
			hr, lr = get_data(self.train_data, self.num_examples_per_epoch_for_train, self.batch_size,
			                  self.num_epoch, shuffle=True, mode=mode, name='%s_data_loader' % mode)
		elif mode == 'valid':
			hr, lr = get_data(self.valid_data, self.num_examples_per_epoch_for_valid, self.valid_bs,
			                  None, shuffle=False, mode=mode, name='%s_data_loader' % mode)
		else:
			raise ValueError('mode must be train or valid')
		
		# split the batch of images and labels for towers
		hr_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=hr)
		lr_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=lr)
		
		return hr_splits, lr_splits
	
	
	def __init__(self):
		# training path
		self.train_data = conf.train_data
		self.models_dir = conf.models_dir
		self.logFilename = conf.log_name
		self.num_examples_per_epoch_for_train = conf.num_train_exps
		
		# for validation
		self.valid_data = conf.valid_data
		self.num_examples_per_epoch_for_valid = conf.num_valid_exps
		
		# make dirs
		if not path.exists(self.models_dir):
			makedirs(self.models_dir)
		
		# soft constraint for total epochs
		self.num_epoch = conf.num_epoch
		
		# device setting
		self.device_id = conf.device_id
		self.num_gpus = conf.num_gpus
		
		# hyper parameters
		self.batch_size = conf.batch_size
		self.valid_bs = conf.valid_bs
		self.weight_decay = conf.weight_decay
		
		# learning rate
		self.lr = tf.placeholder(tf.float32)
		self.base_lr = conf.base_lr
		self.power = conf.power
		self.end_lr = conf.end_lr
		
		# several multiplier
		self.loss_weight = conf.loss_weight
		self.lr_mp = conf.lr_mp
		self.decay_fraction = conf.decay_fraction
		
		# warming-up
		self.warmup_epoch = conf.warmup_epoch
		self.warmup_from0 = conf.warmup_from0		
		
		# resuming and finetune
		self.resume = conf.resume
		self.finetune = conf.finetune
		self.meta_data = conf.meta_data
		
		# whether to enable the non-local block
		self.non_local = conf.non_local		
		
		self.iters = conf.iters
		if self.iters == None:
			if self.resume or self.finetune:
				raise ValueError('iters mush be specified when resume or finetune')
		self.finetune_models_dir = conf.finetune_models_dir
		
		# create an optimizer that performs gradient descent
		opt = tf.train.AdamOptimizer(self.lr)
		
		# get the training dataset
		with tf.device('/cpu:0'):
			t_hr_splits, t_lr_splits = self._get_data(mode='train')
			v_hr_splits, v_lr_splits = self._get_data(mode='valid')
		
		# calculate the gradients for each model tower
		reuse_variables = False
		tower_grads = []
		self.losses = []
		
		# for multi-gpu training
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(self.device_id, self.device_id + self.num_gpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower_%d' % i) as scope:
						# constructs the entire model but shares the variables across all towers
						loss = self._tower_loss(t_hr_splits[i], t_lr_splits[i], i, reuse_variables, self.non_local)
						
						# collect the total losses from each tower
						self.losses += [loss]
						
						# reuse variables for the next tower
						reuse_variables = True
						tf.get_variable_scope().reuse_variables()
						
						# calculate the gradients for the batch of data on this tower
						grads = opt.compute_gradients(loss)
						
						# keep track of the gradients across all towers
						tower_grads.append(grads)
		
		# calculate the mean of each gradient
		# note: this is the synchronization point across all towers
		if self.num_gpus > 1:
			grads = self._average_gradients(tower_grads)
		else:
			grads = tower_grads[0]
		
		# apply the gradients to adjust the shared variables
		self.train_op = opt.apply_gradients(grads)
		
		# for multi-gpu validation
		v_loss = 0.0
		for i in range(self.device_id, self.device_id + self.num_gpus):
			with tf.device('/gpu:%d' % i):
				with tf.name_scope('vtower_%d' % i) as scope:
					net = Net(v_hr_splits[i], v_lr_splits[i], self.non_local, wl=self.weight_decay, tower=i, reuse=True)
					net.build_net()

					v_loss += net.total_loss
		
		self.v_loss = v_loss / self.num_gpus
	
	
	def _init_logging(self):
		logging.basicConfig(
		    level    = logging.DEBUG,
		    #format   = 'LINE %(lineno)-4d  %(levelname)-8s %(message)s',
		    format   = '%(message)s',
		    datefmt  = '%m-%d %H:%M',
		    filename = self.logFilename,
		    filemode = 'w');
		
		# define a Handler which writes INFO messages or higher to the sys.stderr
		console = logging.StreamHandler();
		console.setLevel(logging.DEBUG);
		
		# set a format which is simpler for console use
		#formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s');
		formatter = logging.Formatter('%(message)s');
		# tell the handler to use this format
		console.setFormatter(formatter);
		logging.getLogger('').addHandler(console);
	
	
	def train(self, disp_freq, save_freq, valid_freq, summary_freq=None):
		# initialize logging
		self._init_logging()
		
		# operations for initialization
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver(max_to_keep=int(10e3))
		
		# create a session for running operations in the graph
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable XLA
		sess = tf.Session(config=config)
		
		# initialize the variables (like the epoch counter)
		sess.run(init_op)
		
		# restore trained weights for resuming
		if self.resume or self.finetune:
			saver.restore(sess, path.join(self.finetune_models_dir, 'model.ckpt-' + str(self.iters)))
		
		summary_writer = tf.summary.FileWriter(self.models_dir, sess.graph)
		
		# start input enqueue threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		# global iterations for resuming
		if self.resume:
			iters = self.iters
		
		# for training and finetune
		else:
			iters = 0
		
		# accumulational variables
		sum_time = 0.0
		sum_loss = 0.0
		
		# trace options and metadata
		checkpoint_path = path.join(self.models_dir, 'model.ckpt')
		
		# summary the meta infomation
		if self.meta_data:
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()			
	
			# save iteration 0 and metagraph
			saver.save(sess, checkpoint_path, global_step=iters)
			
			# generate and write a summary
			summary_str = sess.run(summary_op, options=run_options, run_metadata=run_metadata)
			summary_writer.add_run_metadata(run_metadata, 'step%03d' % iters)
			summary_writer.add_summary(summary_str, iters)
		
		# decay policy of learning rate
		decay_fraction_of_epochs = self.decay_fraction
		self.decay_steps = (self.num_examples_per_epoch_for_train * self.num_epoch * decay_fraction_of_epochs) // (self.batch_size)
		
		# calculate warming-up steps
		warmup_steps = self.warmup_epoch * (self.num_examples_per_epoch_for_train // self.batch_size)
		
		# training loop
		try:
			while not coord.should_stop():
				# calculate current learning rate (truncated polynomial decay with warmup)
				if iters < warmup_steps:
					if self.warmup_from0:
						current_lr = self.base_lr * pow(float(iters) / warmup_steps, self.power)
					else:
						current_lr = (self.base_lr - self.end_lr) * pow(float(iters) / warmup_steps, self.power) + self.end_lr
				else:
					if iters <= self.decay_steps:
						current_lr = (self.base_lr - self.end_lr) * pow(1 - float(iters - warmup_steps) / self.decay_steps, self.power) + self.end_lr
					else:
						current_lr = self.end_lr
				
				# apply the multiplier to learning rate
				current_lr = current_lr * self.lr_mp
				
				# run training steps or whatever
				t1 = time()
				_, losses = sess.run([self.train_op, self.losses], feed_dict={self.lr: current_lr})
				t2 = time()
				iters += 1
				
				# normalise the loss
				loss_value = sum(losses) / self.num_gpus
				
				# accumulate
				sum_time += t2 - t1
				sum_loss += loss_value
				
				# display
				if iters % disp_freq == 0:
					logging.info('step %d, loss = %.4f (lr: %.8f, wt: %.2f, time: %.2fs)'
					             % (iters, sum_loss * self.loss_weight / disp_freq, current_lr, self.loss_weight, sum_time))
					
					sum_time = 0.0
					sum_loss = 0.0
				
				# valid
				if iters == 1 or iters % valid_freq == 0:
					v_sum_loss = 0.0
					v_count = 0
					
					print('==valid==')
					vloop = self.num_examples_per_epoch_for_valid // self.valid_bs
					for i in np.arange(0, vloop, 1):
						loss = sess.run(self.v_loss, feed_dict={self.lr: current_lr})
						v_sum_loss += loss
						print('\r%d / %d' % (i+1, vloop), end='')
					print('\r')
					
					logging.info('valid %d, loss = %.4f (wt: %.2f)' % (iters, v_sum_loss * self.loss_weight / vloop, self.loss_weight))
				
				# save a checkpoint
				if iters % save_freq == 0:
					saver.save(sess, checkpoint_path, global_step=iters, write_meta_graph=False)
				
				# write a summary (TBD)
				if summary_freq is not None and iters % summary_freq == -1:
					if self.meta_data:
						summary_str = sess.run(summary_op, options=run_options, run_metadata=run_metadata)
						summary_writer.add_run_metadata(run_metadata, 'step%03d' % iters)
					else:
						summary_str = sess.run(summary_op)
					summary_writer.add_summary(summary_str, iters)
		
		except tf.errors.OutOfRangeError:
			logging.info('Done training -- epoch limit reached')
		finally:
			# when done, ask the threads to stop
			coord.request_stop()
		
		# wait for threads to finish
		coord.join(threads)
		sess.close()
