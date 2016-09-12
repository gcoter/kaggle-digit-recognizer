"""
Useful functions to manage the dataset
"""

from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import os.path
import constants
import pandas as pd

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

def construct_datasets(data_path,pickle_file_path,image_size,max_value,validation_proportion):	
	train_dataset = None 
	train_labels = None 
	valid_dataset = None 
	valid_labels = None 
	test_dataset = None
	""" Read train.csv first """
	train = pd.read_csv(data_path + "train.csv").as_matrix()
	dataset = np.delete(train, 0, axis=1)
	labels_array = train[:,0]
	# Converts each label to one hot vector
	labels = np.ndarray((len(labels_array), num_labels), dtype=np.int32)
	for i in range(len(labels_array)):
		label = labels_array[i]
		labels[i] = one_hot_vector(num_labels,label)
	
	shuffled_dataset, shuffled_labels = randomize(dataset, labels)
	train_dataset, train_labels, valid_dataset, valid_labels = split_with_proportion(shuffled_dataset,shuffled_labels,validation_proportion)
	
	""" Read test.csv """
	test_dataset  = pd.read_csv(data_path + "test.csv").as_matrix()
	
	print('Datasets constructed')
	train_dataset = normalize(train_dataset,max_value)
	valid_dataset = normalize(valid_dataset,max_value)
	test_dataset = normalize(test_dataset,max_value)
	print('Datasets normalized')
	
	save(pickle_file_path,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset)
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset
	
def get_datasets():
	if os.path.isfile(pickle_file_path):
		return load(pickle_file_path)
	else:
		return construct_datasets(data_path,pickle_file_path,image_size,max_pixel_value,validation_proportion)
		
# === VERIFY DATA ===
def row_to_matrix(image_size,row):
	matrix = np.ndarray((image_size,image_size),dtype=row.dtype)
	for px in range(len(row)):
		matrix[px/image_size][px%image_size] = row[px]
	
	return matrix
	
def draw_image(matrix):
	plt.matshow(matrix)
	plt.show()
	
def draw_image_from_row(image_size,row):
	draw_image(row_to_matrix(image_size,row))