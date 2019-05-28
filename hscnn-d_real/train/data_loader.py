"""
Data generator for sr model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def get_data(filename, total_examples, batch_size, num_epoch, shuffle, mode, name='data_loader'):
	with tf.name_scope(name):
		# create a queue that produces the filenames to read
		filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch)
		reader = tf.TFRecordReader()
		
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_example, features={
	        'data': tf.FixedLenFeature([85000], tf.float32)
	    })
		
		data = features['data']
		data = tf.reshape(data, [50, 50, 34])
		
		# augmentation
		if mode == 'train':
			data = tf.image.random_flip_up_down(data)
			data = tf.image.random_flip_left_right(data)
		
		# split
		datas = tf.split(data, [31, 3], axis=2)
		hr = datas[0]
		lr = datas[1]
		
		# ensure that the random shuffling has good mixing properties
		min_fraction_of_examples_in_queue = 0.5
		min_queue_examples = int(total_examples * min_fraction_of_examples_in_queue)
		
		num_preprocess_threads = 24
		if shuffle:
			hr, lr = tf.train.shuffle_batch([hr, lr], batch_size=batch_size,
			                                num_threads=num_preprocess_threads,
			                                capacity=min_queue_examples + 3 * batch_size,
			                                min_after_dequeue=min_queue_examples,
			                                allow_smaller_final_batch=False)
		else:
			hr, lr = tf.train.batch([hr, lr], batch_size=batch_size,
			                        num_threads=num_preprocess_threads,
			                        capacity=min_queue_examples + 3 * batch_size,
			                        allow_smaller_final_batch=False)
		
	return hr, lr
