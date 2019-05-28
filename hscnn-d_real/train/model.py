"""
The core Boosting Network Model
"""

import tensorflow as tf
import numpy as np
from ops import *


class Net(object):
	def __init__(self, hr, lr, non_local, wl, tower, reuse):
		# training inputset
		self.hr = hr
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
			feat0 = self.ddfn(self.lr, 0, b=60)
			feat1 = self.ddfn(self.lr, 1, b=60)
			feat2 = self.ddfn(self.lr, 2, b=60)
			feat = tf.concat([feat0, feat1, feat2], axis=3)
			
			outputs = conv2d(feat, 31, [1, 1], W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
			                add_biases=True, wl=None, reuse=self.reuse, tower_index=self.tower, scope='fusion')
			
			#rrmse_loss = tf.reduce_mean(tf.sqrt(tf.pow(outputs - self.hr, 2)) / self.hr)
			rrmse_loss = tf.reduce_mean(tf.abs(outputs - self.hr) / self.hr)
			self.total_loss = rrmse_loss
