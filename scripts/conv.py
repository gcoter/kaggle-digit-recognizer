from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
from datasetmanagement import get_datasets

import matplotlib.pyplot as plt

# === CONSTANTS ===
image_size = 28
num_labels = 10
data_path = '../data/'
results_path = '../results/'
pickle_file_path = data_path + 'MNIST.pickle'
output_file_path = results_path + 'submission.csv'
validation_proportion = 0.025

# === CONSTRUCT DATASET ===
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset = get_datasets(data_path,pickle_file_path,image_size,validation_proportion)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape)

# === HYPERPARAMETERS ===
patch_size = 5
depths = [1,16,16]
image_size_after_conv = image_size / 4
network_shape = [image_size_after_conv * image_size_after_conv * depths[-1],64,num_labels]

initial_learning_rate = 1E-2
dropout_keep_prob = 0.5

# === MODEL DEFINITION ===
def new_weights(shape):
	return tf.truncated_normal(shape, stddev=0.1)
	
def new_biases(shape):
	return tf.Variable(tf.zeros(shape))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
def max_pool(x,k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')
	
def convolutions(x,image_size,patch_size,depths):
	# Variables
	tf_reshaped_x = tf.reshape(x, [-1,image_size,image_size,1])
	
	## Weights
	weights_conv_1 = new_weights([patch_size,patch_size,depths[0],depths[1]])
	weights_conv_2 = new_weights([patch_size,patch_size,depths[1],depths[2]])
	
	## Biases
	biases_conv_1 = new_biases([depths[1]])
	biases_conv_2 = new_biases([depths[2]])
	
	with tf.name_scope("conv_layer_1"):
		conv_1 = conv2d(tf_reshaped_x, weights_conv_1)
		hidden_1 = tf.nn.relu(conv_1 + biases_conv_1)
		max_pool_1 = max_pool(hidden_1)
		
	with tf.name_scope("conv_layer_2"):
		conv_2 = conv2d(max_pool_1, weights_conv_2)
		hidden_2 = tf.nn.relu(conv_2 + biases_conv_2)
		max_pool_2 = max_pool(hidden_2)
		
	return max_pool_2
	
def mlp(x,network_shape,keep_prob):
	num_layers = len(network_shape)
	
	# Variables
	weights = []
	biases = []

	# Constructs the network according to the given shape array
	for i in range(num_layers-1):
		weights.append(new_weights([network_shape[i], network_shape[i+1]]))
		biases.append(new_biases([network_shape[i+1]]))
		
	# Forward computation (with dropout)
	logits = tf.matmul(x, weights[0]) + biases[0]
	for i in range(1,num_layers-1):
		with tf.name_scope("layer_"+str(i)) as scope:
			logits = tf.matmul(tf.nn.dropout(tf.nn.relu(logits), keep_prob), weights[i]) + biases[i]
			
	return logits

graph = tf.Graph()
with graph.as_default():
	# Input
	tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
	tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

	# Dropout keep probability (set to 1.0 for validation and test)
	keep_prob = tf.placeholder(tf.float32)

	# Global Step for learning rate decay
	global_step = tf.Variable(0)
	
	conv = convolutions(tf_dataset,image_size,patch_size,depths)
	reshaped_conv = tf.reshape(conv, [-1, image_size_after_conv * image_size_after_conv * depths[-1]])
	logits = mlp(reshaped_conv,network_shape,keep_prob)

	# Cross entropy loss
	with tf.name_scope("loss") as scope:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))	
	
	with tf.name_scope("train") as scope:
		learning_rate = initial_learning_rate

		# Passing global_step to minimize() will increment it at each step.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# Predictions for the training, validation, and test data.
	prediction = tf.nn.softmax(logits)
	
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
def plot_results(display_steps, train_points, valid_points):
	plt.plot(display_steps, train_points)
	plt.plot(display_steps, valid_points)
	plt.autoscale(tight=True)
	#plt.xlim(0, display_steps[-1])
	plt.show()
		
def run_training(session, num_steps, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels):
	old_valid_accuracy = None
	weights_values = None
	biases_values = None
	display_steps = []
	train_points = []
	valid_points = []
	tf.initialize_all_variables().run()
	
	summary_writer = tf.train.SummaryWriter(results_path, graph=session.graph)
	
	print('*** Start training',num_epochs,'epochs (',num_steps,'steps) with batch size',batch_size,'***')
	for step in range(num_steps+1):
		# Pick an offset within the training data, which has been randomized.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		_, l, predictions = session.run([optimizer, loss, prediction], feed_dict={tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : dropout_keep_prob})
		
		if (step % display_step == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			minibatch_accuracy = accuracy(session.run(prediction, feed_dict={tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : 1.0}), batch_labels)
			print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)

			valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
			valid_accuracy = accuracy(valid_prediction, valid_labels)
			print("Validation accuracy: %.1f%%" % valid_accuracy)
			
			display_steps.append(step)
			train_points.append(minibatch_accuracy)
			valid_points.append(valid_accuracy)
			
	valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Validation accuracy: %.1f%%" % valid_accuracy)
	
	# PLOT
	plot_results(display_steps, train_points, valid_points)
	
	return weights_values, biases_values
	
# === TRAINING ===
batch_size = 500
num_epochs = 5
display_step = 100
num_steps = 1000 # len(train_dataset)/batch_size * num_epochs

weights_values = None
biases_values = None

with tf.Session(graph=graph) as session:
	weights_values, biases_values = run_training(session, num_steps, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels)
	
	# === TEST ===
	# test_prediction = session.run(prediction, feed_dict={tf_dataset : test_dataset, keep_prob : 1.0})
	# print('Test prediction',test_prediction.shape)

# === GENERATE SUBMISSION FILE ===
def generate_submission_file(test_prediction,output_file_path):
	print('Generating submission file...')
	with open(output_file_path, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['ImageId','Label'])
		print(len(test_prediction))
		for id in range(len(test_prediction)):
			probabilities = test_prediction[id]
			label = np.argmax(probabilities)
			writer.writerow([id+1,label])
		print('Results saved to',output_file_path)
			
# generate_submission_file(test_prediction,output_file_path)