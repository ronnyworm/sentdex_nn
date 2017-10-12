#!/usr/bin/env python3
#coding=UTF-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)

features = 28 * 28

# das kann beliebig sein
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
# der zweite Parameter ist optional und nur als Absicherung, damit TensorFlow das vorher prüft
x = tf.placeholder('float', [None, features])
y = tf.placeholder('float')

def neural_network_model(data):
	# (input_data * weights) + biases

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([features, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(x):
	# output wird ein one-hot vector sein
	prediction = neural_network_model(x)
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
