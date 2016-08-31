from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
from datasetmanagement import get_datasets
import constants
import matplotlib.pyplot as plt

import time

# === CONSTANTS ===
image_size = constants.image_size
max_pixel_value = constants.max_pixel_value
num_labels = constants.num_labels
data_path = constants.data_path
results_path = constants.results_path
pickle_file_path = constants.pickle_file_path
output_file_path = constants.output_file_path
validation_proportion = constants.validation_proportion

# === CONSTRUCT DATASET ===
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset = get_datasets(data_path,pickle_file_path,image_size,max_pixel_value,validation_proportion)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape)

# === HYPERPARAMETERS ===
initial_learning_rate = 1e-4
dropout_keep_prob = 0.5

# === MODEL DEFINITION ===
def new_weights(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def new_biases(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
  
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')
  
def average_pool(x, k=2, stride=2):
	return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')

graph = tf.Graph()
with graph.as_default():
	# Placeholders
	tf_dataset = tf.placeholder(tf.float32, shape=[None, 784])
	tf_labels = tf.placeholder(tf.float32, shape=[None, 10])
	# Dropout keep probability (set to 1.0 for validation and test)
	keep_prob = tf.placeholder(tf.float32)
	
	# (N,28,28,1)
	tf_reshaped_x = tf.reshape(tf_dataset, [-1,image_size,image_size,1])
	
	""" Convolution 1 """
	weights_conv_1 = new_weights([5,5,1,32])
	biases_conv_1 = new_biases([32])
	
	# (N,28,28,32)
	conv_1 = conv2d(tf_reshaped_x, weights_conv_1)
	h_1 = tf.nn.relu(conv_1 + biases_conv_1)
	
	""" Max pool 1 """
	# (N,14,14,32)
	max_pool_1 = max_pool(h_1)
	
	""" Inception 1 """
	# First 1 by 1 convolution...
	weights_conv_0_1x1 = new_weights([1,1,32,8])
	biases_conv_0_1x1 = new_biases([8])
	
	# (N,14,14,8)
	conv_0_1x1 = conv2d(max_pool_1, weights_conv_0_1x1)
	h_conv_0_1x1 = tf.nn.relu(conv_0_1x1 + biases_conv_0_1x1)
	
	# ... followed by other convolutions
	# 5 by 5
	weights_conv_1_5x5 = new_weights([5,5,8,16])
	biases_conv_1_5x5 = new_biases([16])
	
	# (N,14,14,16)
	conv_1_5x5 = conv2d(h_conv_0_1x1, weights_conv_1_5x5)
	h_conv_1_5x5 = tf.nn.relu(conv_1_5x5 + biases_conv_1_5x5)
	
	# 3 by 3
	weights_conv_1_3x3 = new_weights([3,3,8,16])
	biases_conv_1_3x3 = new_biases([16])
	
	# (N,14,14,16)
	conv_1_3x3 = conv2d(h_conv_0_1x1, weights_conv_1_3x3)
	h_conv_1_3x3 = tf.nn.relu(conv_1_3x3 + biases_conv_1_3x3)
	
	# Average pooling followed by 1 by 1 convolution
	# (N,14,14,32)
	avg_pool_0 = average_pool(max_pool_1, stride=1)
	
	weights_conv_1_1x1 = new_weights([1,1,32,24])
	biases_conv_1_1x1 = new_biases([24])
	
	# (N,14,14,24)
	conv_1_1x1 = conv2d(avg_pool_0, weights_conv_1_1x1)
	h_conv_1_1x1 = tf.nn.relu(conv_1_1x1 + biases_conv_1_1x1)
	
	# Concatenation
	# (N,14,14,8+16+16+24=64)
	inception_1 = tf.concat(3, [h_conv_0_1x1, h_conv_1_5x5, h_conv_1_3x3, h_conv_1_1x1])
	
	""" Max pool 2 """
	# (N,7,7,64)
	max_pool_2 = max_pool(inception_1)
	
	# (N,7 * 7 * 64)
	reshaped_conv_output = tf.reshape(max_pool_2, [-1, 7 * 7 * 64])
	
	""" MLP """
	# Variables
	## Weights
	weights_1 = new_weights([7 * 7 * 64, 1024])
	weights_2 = new_weights([1024, 10])
	
	## Biases
	biases_1 = new_biases([1024])
	biases_2 = new_biases([10])
		
	# Forward computation (with dropout)
	logits_1 = tf.matmul(reshaped_conv_output, weights_1) + biases_1
	hidden_1 = tf.nn.relu(logits_1)
	hidden_1_dropout = tf.nn.dropout(hidden_1, keep_prob)
	
	mlp_output = tf.matmul(hidden_1_dropout, weights_2) + biases_2

	prediction = tf.nn.softmax(mlp_output)

	# Train and Evaluate the Model
	loss = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(tf.clip_by_value(prediction,1e-10,1.0)), reduction_indices=[1]))
	optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(loss)

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
	session.run(tf.initialize_all_variables())
	
	time_0 = time.time()
	
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
			minibatch_accuracy = accuracy(session.run(prediction, feed_dict={tf_dataset : batch_data, keep_prob : 1.0}), batch_labels)
			print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)

			valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
			valid_accuracy = accuracy(valid_prediction, valid_labels)
			print("Validation accuracy: %.1f%%" % valid_accuracy)
			
			display_steps.append(step)
			train_points.append(minibatch_accuracy)
			valid_points.append(valid_accuracy)
			
			t = time.time()
			d = t - time_0
			time_0 = t
			
			print("Time :",d,"to compute",display_step,"steps")
			
	valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Validation accuracy: %.1f%%" % valid_accuracy)
	
	# PLOT
	plot_results(display_steps, train_points, valid_points)
	
	return weights_values, biases_values
	
# === TRAINING ===
batch_size = 50
num_epochs = 5
display_step = 100
num_steps = 2500 # len(train_dataset)/batch_size * num_epochs

weights_values = None
biases_values = None

with tf.Session(graph=graph) as session:
	weights_values, biases_values = run_training(session, num_steps, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels)
	
	# === TEST ===
	test_prediction = []
	num_test_steps = len(test_dataset)/batch_size
	print('*** Start testing (',num_test_steps,'steps ) ***')
	for step in range(num_test_steps):
		offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)
		batch_data = test_dataset[offset:(offset + batch_size), :]
		pred = session.run(prediction, feed_dict={tf_dataset : batch_data, keep_prob : 1.0})
		test_prediction.extend(pred)
		
	test_prediction = np.array(test_prediction)
	print('Test prediction',test_prediction.shape)

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
			
generate_submission_file(test_prediction,output_file_path)