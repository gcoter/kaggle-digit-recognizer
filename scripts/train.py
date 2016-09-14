"""
Main code
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv

from datasetmanagement import get_datasets
import constants
import models
import train_ops

# === CONSTANTS ===
output_file_path = constants.output_file_path

# === CONSTRUCT DATASET ===
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset = get_datasets()
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape)

# === DEFINE MODEL ===
model = models.InceptionConvNetV2() # <-- define the model here
	
# === TRAINING ===
batch_size = 50
num_epochs = 10
display_step = 100

with tf.Session(graph=model.graph) as session:
	train_ops.run_training(session, model, num_epochs, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels)
	
	# === TEST ===
	test_prediction = []
	num_test_steps = len(test_dataset)/batch_size
	print('*** Start testing (',num_test_steps,'steps ) ***')
	for step in range(num_test_steps):
		offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)
		batch_data = test_dataset[offset:(offset + batch_size), :]
		pred = session.run(model.prediction, feed_dict={model.batch : batch_data, model.keep_prob : 1.0})
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