from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
from datasetmanagement import get_datasets, randomize
import constants
import matplotlib.pyplot as plt

import time

# === GLOBAL VARIABLE ===
conv_id = 0

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
initial_learning_rate = 1e-2
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
	
def complete_conv2d(x, currentDepth, newDepth, patch_size):
	weights_conv = new_weights([patch_size,patch_size,currentDepth,newDepth])
	biases_conv = new_biases([newDepth])
	conv = tf.nn.conv2d(x, weights_conv, strides=[1, 1, 1, 1], padding='SAME')
	h_conv = tf.nn.relu(conv + biases_conv)
	return h_conv

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
	
	with tf.variable_scope('conv_1'):
		conv_1 = complete_conv2d(tf_reshaped_x,currentDepth=1,newDepth=16,patch_size=5) # (N,28,28,16)
	with tf.variable_scope('max_pool_1'):
		max_pool_1 = max_pool(conv_1) # (N,14,14,32)
	with tf.variable_scope('conv_2'):
		conv_2 = complete_conv2d(max_pool_1,currentDepth=16,newDepth=32,patch_size=5) # (N,28,28,32)
	with tf.variable_scope('inception_1'):
		input = conv_2
		with tf.variable_scope('1x1_branch'):
			with tf.variable_scope('initial_1x1'):
				initial_1x1 = complete_conv2d(input,currentDepth=32,newDepth=8,patch_size=1) # (N,14,14,8)
			with tf.variable_scope('5x5'):
				conv_5x5 = complete_conv2d(initial_1x1,currentDepth=8,newDepth=16,patch_size=5) # (N,14,14,16)
			with tf.variable_scope('3x3'):
				conv_3x3 = complete_conv2d(initial_1x1,currentDepth=8,newDepth=16,patch_size=3) # (N,14,14,16)
		with tf.variable_scope('avg_pool_branch'):
			with tf.variable_scope('initial_avg_pool'):
				initial_avg_pool = average_pool(input, stride=1) # (N,14,14,32)
			with tf.variable_scope('1x1'):
				conv_1x1 = complete_conv2d(initial_avg_pool,currentDepth=32,newDepth=24,patch_size=1) # (N,14,14,24)
		with tf.variable_scope('concatenation'):
			inception = tf.concat(3, [initial_1x1, conv_5x5, conv_3x3, conv_1x1]) # (N,14,14,64)
	with tf.variable_scope('max_pool_2'):
		max_pool_2 = max_pool(inception) # (N,7,7,64)
	
	reshaped_conv_output = tf.reshape(max_pool_2, [-1, 7 * 7 * 64]) # (N,7 * 7 * 64)
	
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

def run_training(session, num_epochs, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels):
	weights_values = None
	biases_values = None
	display_steps = []
	train_points = []
	valid_points = []
	session.run(tf.initialize_all_variables())
	
	time_0 = time.time()
	
	num_steps_per_epoch = len(train_dataset)/batch_size
	num_steps = num_steps_per_epoch * num_epochs
	
	print('*** Start training',num_epochs,'epochs (',num_steps,'steps) with batch size',batch_size,'***')
	for epoch in range(num_epochs):
		print('=== Start epoch',epoch,'===')
		if epoch > 0:
			train_dataset, train_labels = randomize(train_dataset,train_labels)
			print('Randomized dataset and labels for training')
		for step in range(num_steps_per_epoch):
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
batch_size = 100
num_epochs = 5
display_step = 100

weights_values = None
biases_values = None

with tf.Session(graph=graph) as session:
	weights_values, biases_values = run_training(session, num_epochs, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels)
	
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