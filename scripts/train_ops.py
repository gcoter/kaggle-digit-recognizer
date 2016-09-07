from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasetmanagement import randomize

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
	
def plot_results(display_steps, train_points, valid_points):
	plt.plot(display_steps, train_points)
	plt.plot(display_steps, valid_points)
	plt.autoscale(tight=True)
	#plt.xlim(0, display_steps[-1])
	plt.show()

def run_training(session, model, num_epochs, display_step, batch_size, train_dataset, train_labels, valid_dataset, valid_labels):
	display_steps = []
	train_points = []
	valid_points = []
	session.run(tf.initialize_all_variables())
	
	time_0 = time.time()
	
	num_steps_per_epoch = len(train_dataset)/batch_size
	num_steps = num_steps_per_epoch * num_epochs
	step_id = 0
	
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

			_, l, predictions = session.run([model.train_step, model.loss, model.prediction], feed_dict={model.batch : batch_data, model.labels : batch_labels, model.keep_prob : model.dropout_keep_prob})
			
			if (step_id % display_step == 0):
				print("Minibatch loss at step %d: %f" % (step_id, l))
				minibatch_accuracy = accuracy(session.run(model.prediction, feed_dict={model.batch : batch_data, model.keep_prob : 1.0}), batch_labels)
				print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)

				valid_prediction = session.run(model.prediction, feed_dict={model.batch : valid_dataset, model.labels : valid_labels, model.keep_prob : 1.0})
				valid_accuracy = accuracy(valid_prediction, valid_labels)
				print("Validation accuracy: %.1f%%" % valid_accuracy)
				
				display_steps.append(step_id)
				train_points.append(minibatch_accuracy)
				valid_points.append(valid_accuracy)
				
				t = time.time()
				d = t - time_0
				time_0 = t
				
				print("Time :",d,"to compute",display_step,"steps")
				
			step_id += 1
			
	valid_prediction = session.run(model.prediction, feed_dict={model.batch : valid_dataset, model.labels : valid_labels, model.keep_prob : 1.0})
	valid_accuracy = accuracy(valid_prediction, valid_labels)
	print("Validation accuracy: %.1f%%" % valid_accuracy)
	
	# PLOT
	plot_results(display_steps, train_points, valid_points)