from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
from datasetmanagement import get_datasets

import matplotlib.pyplot as plt

# === CONSTANTS ===
image_size = 28
max_pixel_value = 255
num_labels = 10
data_path = '../data/'
results_path = '../results/'
pickle_file_path = data_path + 'MNIST.pickle'
output_file_path = results_path + 'submission.csv'
validation_proportion = 0.025

# === CONSTRUCT DATASET ===
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset = get_datasets(data_path,pickle_file_path,image_size,max_pixel_value,validation_proportion)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape)

print('Training set : mean =',np.mean(train_dataset),"std =",np.std(train_dataset))
print('Validation set : mean =',np.mean(valid_dataset),"std =",np.std(valid_dataset))
print('Test set : mean =',np.mean(test_dataset),"std =",np.std(test_dataset))

# === HYPERPARAMETERS ===
network_shape = [image_size * image_size,1024,num_labels]
num_layers = len(network_shape)
initial_learning_rate = 1E-3
decay_steps = 0
decay_rate = 0.0
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
	summaries = []

	# Constructs the network according to the given shape array
	for i in range(num_layers-1):
		new_weights = tf.Variable(tf.truncated_normal([network_shape[i], network_shape[i+1]],stddev=1.0))
		new_biases = tf.Variable(tf.zeros([network_shape[i+1]]))
		weights.append(new_weights)
		biases.append(new_biases)
		
		summaries.append(tf.histogram_summary("weights_"+str(i), new_weights))
		summaries.append(tf.histogram_summary("biases_"+str(i), new_biases))

	# Global Step for learning rate decay
	global_step = tf.Variable(0)

	# Forward computation (with dropout)
	logits = tf.matmul(tf_dataset, weights[0]) + biases[0]
	for i in range(1,num_layers-2):
		with tf.name_scope("layer_"+str(i)) as scope:
			logits = tf.matmul(tf.nn.dropout(tf.nn.relu(logits), keep_prob), weights[i]) + biases[i]
			
	logits = tf.matmul(logits, weights[-1]) + biases[-1]

	# Cross entropy loss
	with tf.name_scope("loss") as scope:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

		# L2 Regularization
		regularizers = tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(biases[0])
		for i in range(1,num_layers-1):
			regularizers += tf.nn.l2_loss(weights[i]) + tf.nn.l2_loss(biases[i])

		loss += regularization_parameter * regularizers
	
		tf.scalar_summary("loss", loss)
	
	with tf.name_scope("train") as scope:
		learning_rate = initial_learning_rate #tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

		# Passing global_step to minimize() will increment it at each step.
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# Predictions for the training, validation, and test data.
	prediction = tf.nn.softmax(logits)
	
	# Merge all summaries into a single operator
	merged_summary_op = tf.merge_all_summaries()
	
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
def get_parameters(session):
	weights_values = []
	biases_values = []
	for i in range(len(weights)):
		w_values, b_values = session.run([weights[i],biases[i]])
		weights_values.append(w_values)
		biases_values.append(b_values)
		
	return weights_values, biases_values
	
def set_parameters(session, weights_values, biases_values):
	for i in range(len(weights)):
		session.run(weights[i].assign(weights_values[i]))
		session.run(biases[i].assign(biases_values[i]))
	print("Parameters set")

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

		feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : dropout_keep_prob}
		_, l, predictions, merged_summary = session.run([optimizer, loss, prediction, merged_summary_op], feed_dict=feed_dict)
		
		# Write logs for each iteration
		summary_writer.add_summary(merged_summary, step)
			
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
			
			"""
			if old_valid_accuracy is None or valid_accuracy > old_valid_accuracy:
				old_valid_accuracy = valid_accuracy
				weights_values, biases_values = get_parameters(session)
			else:
				print("Early stopping at step",step)
				break			
			"""
			
	valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Validation accuracy: %.1f%%" % valid_accuracy)
	
	# PLOT
	plot_results(display_steps, train_points, valid_points)
	
	return weights_values, biases_values
	
# === TRAINING ===
batch_size = 150
num_epochs = 15
display_step = 100
num_steps = len(train_dataset)/batch_size * num_epochs

weights_values = None
biases_values = None

with tf.Session(graph=graph) as session:
	weights_values, biases_values = run_training(session, num_steps, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels)
	
	# === TEST ===
	test_prediction = session.run(prediction, feed_dict={tf_dataset : test_dataset, keep_prob : 1.0})
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