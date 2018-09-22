"""
I ran this script directly on Kaggle.

It executes only what is necessary to run a model using one Inception Module.

Inception Module greatly improved performances on my CPU while keeping very good accuracy (98.6 %)

My implementation of the Inception Module is inspired from https://www.youtube.com/watch?v=VxhSouuSZDY
"""

from __future__ import print_function
import time
import csv
import pandas as pd
import numpy as np
import tensorflow as tf

# =========
# FUNCTIONS
# =========

def randomize(dataset,labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = np.empty(dataset.shape, dtype=dataset.dtype)
	shuffled_labels = np.empty(labels.shape, dtype=labels.dtype)
	for old_index, new_index in enumerate(permutation):
		shuffled_dataset[new_index] = dataset[old_index]
		shuffled_labels[new_index] = labels[old_index]
	print('Randomized dataset and labels')
	return shuffled_dataset, shuffled_labels
	
def normalize(array,max_value):
	return (array - float(max_value) / 2) / float(max_value)
	
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
	
def new_weights(shape, xavier=True):
	dev = 1.0
	if xavier:
		if len(shape) == 2:
			dev = 1/shape[0]
		elif len(shape) == 4:
			dev = 1/(shape[0]*shape[1]*shape[2])
	initial = tf.random_normal(shape, stddev=dev)
	return tf.Variable(initial)
	
def new_biases(shape, xavier=True):
	dev = 1.0
	if xavier:
		if len(shape) == 2:
			dev = 1/shape[0]
		elif len(shape) == 4:
			dev = 1/(shape[0]*shape[1]*shape[2])
	initial = tf.random_normal(shape, stddev=dev)
	return tf.Variable(initial)

def simple_linear_layer(input,shape):
	assert (len(shape) == 2),"Shape : [input,output]"
	weights = new_weights(shape)
	biases = new_biases([shape[-1]])
	logits = tf.matmul(input, weights) + biases
	return logits
 
def simple_relu_layer(input,shape,dropout_keep_prob=None):
	logits = simple_linear_layer(input,shape)
	logits = tf.nn.relu(logits)
	if not dropout_keep_prob is None:
		logits = tf.nn.dropout(logits, dropout_keep_prob)
	return logits

def conv2d(input, W):
	return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
	
def simple_conv2d(input,currentDepth,newDepth,patch_size):
	weights_conv = new_weights([patch_size,patch_size,currentDepth,newDepth])
	conv = conv2d(input, weights_conv)
	return conv
	
def complete_conv2d(input, currentDepth, newDepth, patch_size):
	weights_conv = new_weights([patch_size,patch_size,currentDepth,newDepth])
	biases_conv = new_biases([newDepth])
	conv = conv2d(input, weights_conv)
	h_conv = tf.nn.relu(conv + biases_conv)
	return h_conv

def max_pool(input, k=2):
	return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')
  
def average_pool(input, k=2, stride=2):
	return tf.nn.avg_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')

def seconds2minutes(time):
	minutes = int(time) / 60
	seconds = int(time) % 60
	return minutes, seconds

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
# =========
#   CODE
# =========

# === CONSTANTS ===
image_size = 28
max_pixel_value = 255
num_labels = 10
validation_proportion = 0.1
data_path = '../data/' # The competition datafiles are in the directory ../input
results_path = '../results/'
output_file_path = results_path + 'submission.csv'
parameters_path = "../parameters/"
model_path = parameters_path + "model.ckpt"
display_step = 100

# === HYPERPARAMETERS ===
initial_learning_rate = 1e-3
dropout_keep_prob = 0.5
batch_size = 100
num_epochs = 25

# === CONSTRUCT DATASET ===
""" Read train.csv first """
train = pd.read_csv(data_path + "train.csv").as_matrix()
dataset = np.delete(train, 0, axis=1)
labels_array = train[:,0]
# Convert each label to one hot vector
labels = np.ndarray((len(labels_array), num_labels), dtype=np.int32)
for i in range(len(labels_array)):
	label = labels_array[i]
	labels[i] = one_hot_vector(num_labels,label)

# Randomize dataset and labels
shuffled_dataset, shuffled_labels = randomize(dataset, labels)
train_dataset, train_labels, valid_dataset, valid_labels = split_with_proportion(shuffled_dataset,shuffled_labels,validation_proportion)

""" Read test.csv """
test_dataset  = pd.read_csv(data_path + "test.csv").as_matrix()

print('Datasets constructed')

# Normalize datasets
train_dataset = normalize(train_dataset,max_pixel_value)
valid_dataset = normalize(valid_dataset,max_pixel_value)
test_dataset = normalize(test_dataset,max_pixel_value)
print('Datasets normalized')

# === DEFINE MODEL ===
graph = tf.Graph()
with graph.as_default():
	# Input
	batch = tf.placeholder(tf.float32, shape=(None, image_size*image_size))
	labels = tf.placeholder(tf.float32, shape=(None, num_labels))

	# Dropout keep probability (set to 1.0 for validation and test)
	keep_prob = tf.placeholder(tf.float32)

	# Forward computation
	reshaped_input = tf.reshape(batch, [-1,image_size,image_size,1]) # (N,28,28,1)		
	
	with tf.variable_scope('conv_1'):
		conv_1 = complete_conv2d(reshaped_input,currentDepth=1,newDepth=32,patch_size=3) # (N,28,28,32)
	with tf.variable_scope('max_pool_1'):
		max_pool_1 = max_pool(conv_1) # (N,14,14,32)
	with tf.variable_scope('conv_2'):
		conv_2 = complete_conv2d(max_pool_1,currentDepth=32,newDepth=64,patch_size=3) # (N,14,14,64)
	with tf.variable_scope('max_pool_2'):
		max_pool_2 = max_pool(conv_2) # (N,7,7,64)
		
	image_size_after_conv = image_size//4
		
	reshaped_conv_output = tf.reshape(max_pool_2, [-1, image_size_after_conv*image_size_after_conv*64]) # (N,7*7*64)
	hidden_1 = simple_relu_layer(reshaped_conv_output, shape=[image_size_after_conv*image_size_after_conv*64,1000],dropout_keep_prob=dropout_keep_prob)
	logits_out = simple_linear_layer(hidden_1, shape=[1000,num_labels])

	# Cross entropy loss
	with tf.name_scope("loss") as scope:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_out, labels))
	
	with tf.name_scope("train_step") as scope:
		train_step = tf.train.AdamOptimizer(initial_learning_rate).minimize(loss)

	# Predictions for the training, validation, and test data.
	prediction = tf.nn.softmax(logits_out)
	
def test_on_valid_data(session,valid_dataset,valid_labels,batch_size):
	num_valid_examples = len(valid_dataset)
	num_steps = num_valid_examples//batch_size
	avg_accuracy = 0.0
	for step in range(num_steps):
		batch_data = valid_dataset[step * batch_size:(step + 1) * batch_size]
		batch_labels = valid_labels[step * batch_size:(step + 1) * batch_size]
		minibatch_accuracy = accuracy(session.run(prediction, feed_dict={batch : batch_data, keep_prob : 1.0}), batch_labels)
		avg_accuracy += minibatch_accuracy
	return avg_accuracy/num_steps
	
# === TRAINING ===
with tf.Session(graph=graph) as session:
	saver = tf.train.Saver()
	session.run(tf.global_variables_initializer())
	
	total_time = 0.0
	begin_time = time_0 = time.time()
	
	num_steps_per_epoch = len(train_dataset)//batch_size
	num_steps = num_steps_per_epoch * num_epochs
	step_id = 0
	
	valid_accuracy_max = 0.0
	
	print('*** Start training',num_epochs,'epochs (',num_steps,'steps) with batch size',batch_size,'***')
	for epoch in range(num_epochs):
		print('=== Start epoch',epoch,'===')
		if epoch > 0:
			train_dataset, train_labels = randomize(train_dataset,train_labels)
		for step in range(num_steps_per_epoch):
			# Pick an offset within the training data, which has been randomized
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			# Generate a minibatch
			batch_data = train_dataset[offset:(offset + batch_size), :]
			batch_labels = train_labels[offset:(offset + batch_size), :]

			# Train step
			_, l = session.run([train_step, loss], feed_dict={batch : batch_data, labels : batch_labels, keep_prob : dropout_keep_prob})
			
			if (step_id % display_step == 0):
				# Calculate minibatch accuracy
				print("Minibatch loss at step %d: %f" % (step_id, l))
				minibatch_accuracy = accuracy(session.run(prediction, feed_dict={batch : batch_data, keep_prob : 1.0}), batch_labels)
				print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)

				# Calculate accuracy on validation set
				valid_accuracy = test_on_valid_data(session,valid_dataset,valid_labels,batch_size)
				print("Validation accuracy: %.1f%%" % valid_accuracy)
				
				if valid_accuracy > valid_accuracy_max:
					saver.save(session, model_path)
					valid_accuracy_max = valid_accuracy
					print("Parameters saved to",model_path)
				
				# Time spent is measured
				t = time.time()
				d = t - time_0
				time_0 = t
				
				print("Time :",d,"s to compute",display_step,"steps")
				
			step_id += 1
	
	total_time = time.time() - begin_time
	
	total_time_minutes, total_time_seconds = seconds2minutes(total_time)
	
	saver.restore(session, model_path)
	print("Parameters restored from",model_path,"( valid_accuracy_max =",valid_accuracy_max,")")
	
	valid_prediction = session.run(prediction, feed_dict={batch : valid_dataset, labels : valid_labels, keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Validation accuracy: %.1f%%" % valid_accuracy)
	
	print("*** Total time :",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")
	
	# === TEST ===
	test_prediction = []
	num_test_steps = len(test_dataset)//batch_size
	print('*** Start testing (',num_test_steps,'steps ) ***')
	for step in range(num_test_steps):
		offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)
		batch_data = test_dataset[offset:(offset + batch_size), :]
		pred = session.run(prediction, feed_dict={batch : batch_data, keep_prob : 1.0})
		test_prediction.extend(pred)
		
	test_prediction = np.array(test_prediction)
	print('Test prediction',test_prediction.shape)
	
# === GENERATE SUBMISSION FILE ===
print('Generating submission file...')
with open(output_file_path, 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['ImageId','Label'])
	print(len(test_prediction))
	for id in range(len(test_prediction)):
		probabilities = test_prediction[id]
		label = np.argmax(probabilities)
		writer.writerow([id+1,label])
	print('Results saved to',output_file_path)