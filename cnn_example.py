#!/usr/bin/env python3
#coding=UTF-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)

features = 28 * 28

n_classes = 10
batch_size = 128

use_dropout = True
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# height x width
# der zweite Parameter ist optional und nur als Absicherung, damit TensorFlow das vorher prüft
x = tf.placeholder('float', [None, features])
y = tf.placeholder('float')

def conv2d(x, W):
	# strides: movement
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	# ksize: size of window
	# strides: movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
	# 5 by 5 convolution, 1 input, produces 32 features / outputs
	weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
				'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
				# fully connected, not a convolution any more
				'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
				'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	# immer nur für output
	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
				'b_conv2': tf.Variable(tf.random_normal([64])),
				# fully connected, not a convolution any more
				'b_fc': tf.Variable(tf.random_normal([1024])),
				'out': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	# keep_rate % of our neurons will be kept
	# in a much larger dataset, dropout would make a difference
	if use_dropout:
		fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']

	return output


def train_neural_network(x):
	# output wird ein one-hot vector sein
	prediction = convolutional_neural_network(x)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# AdamOptimizer hat eine default learning rate von 0.001, die wir hier verwenden
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	# cycles feed forward + backprop
	epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				i, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
