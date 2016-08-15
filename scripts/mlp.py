from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv

# === CONSTRUCT DATASETS ===
image_size = 28
num_labels = 10

def contruct_datasets():
	return [], [], [], []

train_dataset, train_labels, valid_dataset, valid_labels = contruct_datasets()

# === HYPERPARAMETERS ===
""" With those parameters, I get 86.7% of accuracy """
network_shape = [image_size * image_size,600,300,num_labels]
initial_learning_rate = 0.1
decay_steps = 0
decay_rate = 0
regularization_parameter = 0.0
dropout_keep_prob = 0.5

# === MODEL DEFINITION ===
graph = tf.Graph()
with graph.as_default():
	# Input
	tf_dataset = tf.placeholder(tf.float32, shape=(None, network_shape[0]))
	tf_labels = tf.placeholder(tf.float32, shape=(None, network_shape[-1]))

	# Dropout keep probability (set to 1.0 for validation and test)
	keep_prob = tf.placeholder(tf.float32)

	# Variables
	weights = []
	biases = []

	# Constructs the network according to the given shape array
	for i in range(num_layers-1):
		weights.append(tf.Variable(tf.truncated_normal([network_shape[i], network_shape[i+1]],stddev=1.0/(network_shape[i]))))
		biases.append(tf.Variable(tf.zeros([network_shape[i+1]])))

	# Global Step for learning rate decay
	global_step = tf.Variable(0)

	# Training computation (with dropout)
	logits = tf.matmul(tf_dataset, weights[0]) + biases[0]
	for i in range(1,num_layers-1):
		logits = tf.matmul(tf.nn.dropout(tf.nn.relu(logits), keep_prob), weights[i]) + biases[i]

	# Cross entropy loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

	# L2 Regularization
	regularizers = tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(biases[0])
	for i in range(1,num_layers-1):
		regularizers += tf.nn.l2_loss(weights[i]) + tf.nn.l2_loss(biases[i])

	loss += regularization_parameter * regularizers

	learning_rate = initial_learning_rate #tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

	# Passing global_step to minimize() will increment it at each step.
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# Predictions for the training, validation, and test data.
	prediction = tf.nn.softmax(logits)
	
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
# === TRAINING ===
old_valid_accuracy = None
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized with shape ", str(network_shape))
	for step in range(num_steps+1):
		# Pick an offset within the training data, which has been randomized.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : dropout_keep_prob}
		_, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
			
		if (step % 100 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(session.run(prediction, feed_dict={tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : 1.0}), batch_labels))

			valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
			valid_accuracy = accuracy(valid_prediction, valid_labels)
			print("Validation accuracy: %.1f%%" % valid_accuracy)
			old_valid_accuracy = valid_accuracy
			
# === TEST ===
test_dataset = []
test_prediction = session.run(prediction, feed_dict={tf_dataset : test_dataset, tf_labels : test_labels, keep_prob : 1.0})

# === GENERATE SUBMISSION FILE ===
def generate_submission_file(test_prediction):
	print("TO DO")