from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
import os.path
from six.moves import cPickle as pickle

# === CONSTRUCT DATASETS ===
image_size = 28
num_labels = 10
data_path = '../data/'
results_path = '../results/'
pickle_file_path = data_path + 'MNIST.pickle'
output_file_path = results_path + 'submission.csv'
validation_proportion = 0.1

def initialize_dataset_array(num_rows,image_size):
	return np.ndarray((num_rows, image_size * image_size), dtype=np.uint8)

def initialize_array(num_rows,image_size,num_labels):
	dataset = initialize_dataset_array(num_rows,image_size)
	labels = np.ndarray((num_rows, num_labels), dtype=np.int32)
	return dataset, labels
	
def randomize(dataset,labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:]
	shuffled_labels = labels[permutation]
	print('Randomized dataset and labels')
	return shuffled_dataset, shuffled_labels
	
def split_with_proportion(shuffled_dataset,shuffled_labels,validation_proportion):
	validation_index = int(len(shuffled_dataset) * validation_proportion)
	train_dataset = shuffled_dataset[validation_index:]
	train_labels = shuffled_labels[validation_index:]
	valid_dataset = shuffled_dataset[:validation_index]
	valid_labels = shuffled_labels[:validation_index]
	return train_dataset, train_labels, valid_dataset, valid_labels
	
def one_hot_vector(num_labels,label):
	vector = np.zeros(num_labels, dtype=np.int32)
	vector[int(label)] = 1
	return vector
	
def save(pickle_file_path,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset):
	try:
		f = open(pickle_file_path, 'wb')
		save = {
			'train_dataset': train_dataset,
			'train_labels': train_labels,
			'valid_dataset': valid_dataset,
			'valid_labels': valid_labels,
			'test_dataset': test_dataset
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		print('Datasets saved to ', pickle_file_path)
	except Exception as e:
		print('Unable to save data to', pickle_file_path, ':', e)
		raise
		
def load(pickle_file_path):
	with open(pickle_file_path, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		valid_dataset = save['valid_dataset']
		valid_labels = save['valid_labels']
		test_dataset = save['test_dataset']
		del save  # hint to help gc free up memory
		print('Loaded datasets from ', pickle_file_path)
	  
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset

def construct_datasets(data_path,pickle_file_path,image_size,validation_proportion):	
	train_dataset = None 
	train_labels = None 
	valid_dataset = None 
	valid_labels = None 
	test_dataset = None
	""" Read train.csv first """
	with open(data_path + 'train.csv', 'rb') as csvfile:
		print('Reading train csv file...')
		reader = csv.reader(csvfile, delimiter=',')
		num_rows = sum(1 for row in reader)
		num_images = num_rows - 1
		csvfile.seek(0)
		print(num_rows,'rows counted')
		dataset, labels = initialize_array(num_images,image_size,num_labels)
		print('Data set', dataset.shape, labels.shape)
		id = 0
		for row in reader:
			# skip first row
			if id > 0:
				dataset[id-1] = row[1:]
				labels[id-1] = one_hot_vector(num_labels,row[0])
			id += 1
		print(id, 'rows read')
	shuffled_dataset, shuffled_labels = randomize(dataset, labels)
	train_dataset, train_labels, valid_dataset, valid_labels = split_with_proportion(shuffled_dataset,shuffled_labels,validation_proportion)
	
	""" Read test.csv """
	with open(data_path + 'test.csv', 'rb') as csvfile:
		print('Reading test csv file...')
		reader = csv.reader(csvfile, delimiter=',')
		num_rows = sum(1 for row in reader)
		num_images = num_rows - 1
		csvfile.seek(0)
		print(num_rows,'rows counted')
		test_dataset = initialize_dataset_array(num_images,image_size)
		print('Test Data set', test_dataset.shape)
		id = 0
		for row in reader:
			# skip first row
			if id > 0:
				test_dataset[id-1] = row[0:]
			id += 1
		print(id, 'rows read')
	
	print('Datasets constructed')
	save(pickle_file_path,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset)
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset
	
def get_datasets(data_path,pickle_file_path,image_size,validation_proportion):
	if os.path.isfile(pickle_file_path):
		return load(pickle_file_path)
	else:
		return construct_datasets(data_path,pickle_file_path,image_size,validation_proportion)

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset = get_datasets(data_path,pickle_file_path,image_size,validation_proportion)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape)

# === HYPERPARAMETERS ===
network_shape = [image_size * image_size,1024,num_labels]
num_layers = len(network_shape)
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
batch_size = 150
num_epochs = 1
num_steps = len(train_dataset)/batch_size * num_epochs
old_valid_accuracy = None
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized with shape ", str(network_shape))
	print('Start training',num_epochs,'epochs (',num_steps,'steps) with batch size',batch_size)
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