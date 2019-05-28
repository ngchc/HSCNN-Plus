"""
Various tensorflow operators
conv2d: NCHW->NHWC
"""

import tensorflow as tf
import numpy as np


def _atrous_conv2d(value, filters, rate, padding, name=None):
	return tf.nn.convolution(
		input=value,
		filter=filters,
		padding=padding,
		dilation_rate=np.broadcast_to(rate, (2,)),
	    data_format='NHWC',
	    name=name)


def conv2d(inputs, num_outputs, kernel_shape=[3, 3], strides=[1, 1], add_biases=True, pad='SAME', dilated=1, reuse=False, tower_index=None,
           W_init=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), # msra
           b_init=tf.constant_initializer(0.0), W_params=None, b_params=None, wl=None, wl_type=tf.nn.l2_loss, summary=False, scope='conv2d'):
	"""
	Args:
	  inputs: NHWC
	  num_outputs: the number of filters
	  kernel_shape: [height, width]
	  strides: [height, width]
	  pad: 'SAME' or 'VALID'
	  W/b_params: lists for layer-wise learning rate and gradient clipping
	  wl: add weight losses to collection
	  reuse: reusage of variables
	  dilated: convolution with holes
	
	Returns:
	  outputs: NHWC
	"""
	with tf.variable_scope(scope, reuse=reuse):
		# get shapes
		kernel_h, kernel_w = kernel_shape
		stride_h, stride_w = strides
		batch_size, height, width, in_channel = inputs.get_shape().as_list()
		
		weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
		weights = tf.get_variable('w', weights_shape, tf.float32, W_init)
		
		# add summary for w
		if summary and not reuse:
			tf.summary.histogram('hist_w', weights)
		
		# add to the list of weights
		if W_params is not None and not reuse:
			W_params += [weights]
		
		# 2-D convolution
		if dilated != 1:
			outputs = _atrous_conv2d(inputs, weights, rate=dilated, padding=pad)
		else:
			outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=pad, data_format='NHWC')
		
		# add biases
		if add_biases:
			biases = tf.get_variable('b', [num_outputs], tf.float32, b_init)
			
			# add summary for b
			if summary and not reuse:
				tf.summary.histogram('hist_b', biases)
				
			# add to the list of biases
			if b_params is not None and not reuse:
				b_params += [biases]
			
			outputs = tf.nn.bias_add(outputs, biases, data_format='NHWC')
		
		# add weight decay
		if wl is not None:
			weight_loss = tf.multiply(wl_type(weights), wl, name='weight_loss')
			tf.add_to_collection('losses_' + str(tower_index), weight_loss)
		
		return outputs
