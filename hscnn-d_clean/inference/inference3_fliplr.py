import tensorflow as tf
import numpy as np

import os
import h5py
from PIL import Image
from ops import *
import scipy.io as sio


def modcrop(im, modulo):
	if len(im.shape) == 3:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1], :]
	elif len(im.shape) == 2:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1]]
	else:
		raise AttributeError
	return im


def shave(im, border):
	if len(im.shape) == 3:
		return im[border[0] : -border[0], 
		          border[1] : -border[1], :]
	elif len(im.shape) == 2:
		return im[border[0] : -border[0], 
		          border[1] : -border[1]]
	else:
		raise AttributeError


class Net(object):
	def __init__(self, lr, non_local, wl, tower, reuse):
		# training inputset
		self.lr = lr
		
		# multi-gpu related settings
		self.reuse = reuse
		self.tower = tower
		
		# parameter lists for weights and biases
		self.W_params = []
		self.b_params = []
		
		# coefficient of weight decay
		self.wl = wl
		
		# whether to enable the non-local block
		self.non_local = non_local
	
	
	def dfus_block(self, bottom, i):
		act = tf.nn.relu

		with tf.variable_scope('dfus_block' + str(i), reuse=self.reuse):
			conv1  = act(conv2d(bottom, 64, [1, 1], wl=None, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_i'), name='relu' + str(i) + '_i')

			feat1  = act(conv2d(conv1, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_1'), name='relu' + str(i) + '_1')
			feat15 = act(conv2d(feat1, 8, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_15'), name='relu' + str(i) + '_15')

			feat2  = act(conv2d(conv1, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_2'), name='relu' + str(i) + '_2')
			feat23 = act(conv2d(feat2, 8, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_23'), name='relu' + str(i) + '_23')

			feat = tf.concat([feat1, feat15, feat2, feat23], 3, name='conv' + str(i) + '_c1')
			feat = act(conv2d(feat, 16, [1, 1], wl=None, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_r'), name='relu' + str(i) + '_r')

			top = tf.concat([bottom, feat], 3, name='conv' + str(i) + '_c2')

		return top


	def ddfn(self, bottom, step, b=10):
		act = tf.nn.relu

		with tf.variable_scope('ddfn_' + str(step), reuse=self.reuse):
			with tf.name_scope('msfeat'):
				conv13  = act(conv2d(bottom, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_3'), name='relu1_3')
				conv15  = act(conv2d(bottom, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_5'), name='relu1_5')

				conv135 = act(conv2d(conv13, 16, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_3_5'), name='relu1_3_5')
				conv153 = act(conv2d(conv15, 16, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_5_3'), name='relu1_5_3')

				conv1 = tf.concat([conv13, conv15, conv135, conv153], 3, name='conv1_c')

			if self.non_local:
				conv1, _ = non_local_block(conv1, reuse=self.reuse, tower_index=self.tower)

			feat = self.dfus_block(conv1, 2)

			for i in range(3, b, 1):
				feat = self.dfus_block(feat, i)

			top = feat

			return top
	
	
	def build_net(self):
		with tf.variable_scope('net', reuse=self.reuse):
			feat0 = self.ddfn(self.lr, 0, b=40)
			feat1 = self.ddfn(self.lr, 1, b=40)
			feat2 = self.ddfn(self.lr, 2, b=40)
			feat = tf.concat([feat0, feat1, feat2], axis=3)
			
			outputs = conv2d(feat, 31, [1, 1], W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
			                add_biases=True, wl=None, reuse=self.reuse, tower_index=self.tower, scope='fusion')
			self.sr = outputs


def main():
	# folder path
	lr_folder = './NTIRE2018_Test_Clean'
	output_folder = './test_results_3'
	test_ims = ['Fri_1215-1050', 'Fri_1215-1221', 'Sat_1223-1151', 'Sat_1223-1321', 'Sat_1223-1550']
	
	with tf.device('/cpu:0'):
		lr = tf.placeholder('float', [1, None, None, 3])
	
	# recreate the network
	net = Net(lr, False, None, 0, reuse=False)
	with tf.device('/cpu:0'):
		net.build_net()
	output = net.sr
	
	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	# restore weights
	saver = tf.train.Saver()
	
	# set a range for searching
	start_iter = 225600
	step_iter = 1000
	end_iter = 225600
	
	# search for the best model
	record_psnr = []
	for itera in np.arange(start_iter, end_iter + step_iter, step_iter):
		saver.restore(sess, os.path.join('./models', 'model.ckpt-' + str(itera)))
		for name in test_ims:
			im_name = name + '_clean.png'
			im_lr = np.array(Image.open(os.path.join(lr_folder, im_name)))
			im_lr = im_lr.astype(np.float32) / 255.0

			im_lr = np.fliplr(im_lr)
			im_lr = np.expand_dims(im_lr, axis=0)

			im_sr = sess.run(output, feed_dict={lr: im_lr})
			im_sr = np.squeeze(im_sr) * 4095.0
			
			im_sr = np.fliplr(im_sr)

			mat_name = name + '_ex.mat'
			sio.savemat(os.path.join(output_folder, mat_name), {'rad': np.transpose(im_sr, [0, 1, 2])})
			print(name)


if __name__ == '__main__':
	main()
