#!/usr/bin/env python3
#coding=UTF-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# https://stackoverflow.com/questions/42311007/attributeerror-tensorflow-python-ops-rnn-has-no-attribute-rnn
from tensorflow.contrib import rnn 

mnist = input_data.read_data_sets("mnist/", one_hot=True)

epochs = 3
n_classes = 10
batch_size = 100
chunk_size = 28
chunk_count = 28
rnn_size = 128

# height x width
# der zweite Parameter ist optional und nur als Absicherung, damit TensorFlow das vorher prüft
x = tf.placeholder('float', [None, chunk_count, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
	# (input_data * weights) + biases

	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	# tf.split has changed arguments order (https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/RELEASE.md):
	x = tf.split(x, chunk_count, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output


def train_neural_network(x):
	# output wird ein one-hot vector sein
	prediction = recurrent_neural_network(x)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# AdamOptimizer hat eine default learning rate von 0.001, die wir hier verwenden
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, chunk_count, chunk_size))

				i, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, chunk_count, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
