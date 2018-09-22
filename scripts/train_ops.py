"""
Operations useful for training
"""

from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasetmanagement import randomize

def seconds2minutes(time):
	minutes = int(time) / 60
	seconds = int(time) % 60
	return minutes, seconds

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
def plot_results(display_steps, train_points, valid_points):
	plt.plot(display_steps, train_points)
	plt.plot(display_steps, valid_points)
	plt.autoscale(tight=True)
	#plt.xlim(0, display_steps[-1])
	plt.show()

""" Main method for training """
def run_training(session, model, num_epochs, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels):
	display_steps = []
	train_points = []
	valid_points = []
	session.run(tf.initialize_all_variables())
	
	total_time = 0.0
	begin_time = time_0 = time.time()
	
	num_steps_per_epoch = len(train_dataset)//batch_size
	num_steps = num_steps_per_epoch * num_epochs
	step_id = 0
	
	save_path = '../parameters/model.ckpt'
	valid_accuracy_max = {'value' : 0.0, 'step' : 0}
	
	print('*** Start training',num_epochs,'epochs (',num_steps,'steps) with batch size',batch_size,'***')
	for epoch in range(num_epochs):
		print('=== Start epoch',epoch,'===')
		if epoch > 0:
			train_dataset, train_labels = randomize(train_dataset,train_labels)
		for step in range(num_steps_per_epoch):
			# Pick an offset within the training data, which has been randomized.
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			# Generate a minibatch.
			batch_data = train_dataset[offset:(offset + batch_size), :]
			batch_labels = train_labels[offset:(offset + batch_size), :]

			# Train step
			_, l = session.run([model.train_step, model.loss], feed_dict={model.batch : batch_data, model.labels : batch_labels, model.keep_prob : model.dropout_keep_prob})
			
			if (step_id % display_step == 0):
				# Calculate minibatch accuracy
				print("Minibatch loss at step %d: %f" % (step_id, l))
				minibatch_accuracy = accuracy(session.run(model.prediction, feed_dict={model.batch : batch_data, model.keep_prob : 1.0}), batch_labels)
				print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)

				# Calculate accuracy on validation set
				valid_prediction = session.run(model.prediction, feed_dict={model.batch : valid_dataset, model.labels : valid_labels, model.keep_prob : 1.0})
				valid_accuracy = accuracy(valid_prediction, valid_labels)
				print("Validation accuracy: %.1f%%" % valid_accuracy)
				
				# Create a checkpoint when a new max is reached
				if valid_accuracy > valid_accuracy_max['value']:
					model.saver.save(session, save_path)
					valid_accuracy_max['value'] = valid_accuracy
					valid_accuracy_max['step'] = step_id
					print("New max value:",valid_accuracy_max['value'],"%")
					print("Model saved to",save_path)
				
				# For plotting
				display_steps.append(step_id)
				train_points.append(minibatch_accuracy)
				valid_points.append(valid_accuracy)
				
				# Time spent is measured
				t = time.time()
				d = t - time_0
				time_0 = t
				
				print("Time :",d,"s to compute",display_step,"steps")
				
			step_id += 1
	
	total_time = time.time() - begin_time
	
	total_time_minutes, total_time_seconds = seconds2minutes(total_time)
	
	# Restore from checkpoint (parameters that give the best accuracy)
	model.saver.restore(session, save_path)
	print("Model restored from step",valid_accuracy_max['step'],"(expected accuracy:",valid_accuracy_max['value'],"%)")
	
	valid_prediction = session.run(model.prediction, feed_dict={model.batch : valid_dataset, model.labels : valid_labels, model.keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Final validation accuracy: %.1f%%" % valid_accuracy)
	
	print("*** Total time :",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")
	
	# PLOT
	plot_results(display_steps, train_points, valid_points)