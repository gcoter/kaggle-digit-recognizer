"""
Models definitions
"""

import abc
import tensorflow as tf

import constants
import layers

# === CONSTANTS ===
image_size = constants.image_size
num_labels = constants.num_labels

# === HELPING FUNCTIONS ===
def get_mean_variance(input):
	mean, variance = tf.nn.moments(input,axes=[0,1,2,3])
	
	return mean, variance
	
# === MODELS ===
"""
AbstractModel generalizes the construction of a model.

Subclasses implements the only variable part.
"""
class AbstractModel(object):
	def __init__(self):
		return
	
	def construct(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			# Input
			self.batch = tf.placeholder(tf.float32, shape=(None, image_size*image_size))
			self.labels = tf.placeholder(tf.float32, shape=(None, num_labels))

			# Dropout keep probability (set to 1.0 for validation and test)
			self.keep_prob = tf.placeholder(tf.float32)

			# Forward computation (model has to be defined in subclasses)
			self.logits_out = self.model(self.batch) # <-- this part is defined by subclasses

			# Cross entropy loss
			with tf.name_scope("loss") as scope:
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_out, self.labels))
			
			with tf.name_scope("train_step") as scope:
				self.train_step = tf.train.AdamOptimizer(self.initial_learning_rate).minimize(self.loss)

			# Predictions for the training, validation, and test data.
			self.prediction = tf.nn.softmax(self.logits_out)
			
			# Add ops to save and restore all the variables.
			self.saver = tf.train.Saver()

	@abc.abstractmethod
	def model(self,input):
		""" Model definition here (must return logits) """
		return

class MLP(AbstractModel):
	def __init__(self):
		super(MLP,self).__init__()
		self.initial_learning_rate = 1e-3
		self.dropout_keep_prob = 0.5
		self.construct()
		
	def model(self,input):
		self.hidden = layers.simple_relu_layer(input, shape=[image_size*image_size,1024],dropout_keep_prob=self.dropout_keep_prob)
		return layers.simple_linear_layer(self.hidden, shape=[1024,num_labels])
			
class SimpleConvNet(AbstractModel):
	def __init__(self):
		super(SimpleConvNet,self).__init__()
		self.patch_size = 5
		self.initial_learning_rate = 1e-3
		self.dropout_keep_prob = 0.5
		self.construct()
		
	def model(self,input):
		self.reshaped_input = tf.reshape(input, [-1,image_size,image_size,1]) # (N,28,28,1)
		
		with tf.variable_scope('conv_1'):
			self.conv_1 = layers.complete_conv2d(self.reshaped_input,currentDepth=1,newDepth=32,patch_size=self.patch_size) # (N,28,28,32)
		with tf.variable_scope('max_pool_1'):
			self.max_pool_1 = layers.max_pool(self.conv_1) # (N,14,14,32)
		with tf.variable_scope('conv_2'):
			self.conv_2 = layers.complete_conv2d(self.max_pool_1,currentDepth=32,newDepth=64,patch_size=self.patch_size) # (N,14,14,64)
		with tf.variable_scope('max_pool_2'):
			self.max_pool_2 = layers.max_pool(self.conv_2) # (N,7,7,64)
		
		image_size_after_conv = image_size/4
		
		self.reshaped_conv_output = tf.reshape(self.max_pool_2, [-1, image_size_after_conv*image_size_after_conv*64])
		self.hidden = layers.simple_relu_layer(self.reshaped_conv_output, shape=[image_size_after_conv*image_size_after_conv*64,1024],dropout_keep_prob=self.dropout_keep_prob)
		return layers.simple_linear_layer(self.hidden, shape=[1024,num_labels])
		
class InceptionConvNetV1(AbstractModel):
	def __init__(self):
		super(InceptionConvNetV1,self).__init__()
		self.initial_learning_rate = 1e-3
		self.dropout_keep_prob = 0.5
		self.construct()
		
	def model(self,input):
		self.reshaped_input = tf.reshape(input, [-1,image_size,image_size,1]) # (N,28,28,1)
		
		with tf.variable_scope('conv_1'):
			self.conv_1 = layers.complete_conv2d(self.reshaped_input,currentDepth=1,newDepth=32,patch_size=5) # (N,28,28,32)
		with tf.variable_scope('max_pool_1'):
			self.max_pool_1 = layers.max_pool(self.conv_1) # (N,14,14,32)
		""" Inception module """
		with tf.variable_scope('inception_1'):
			self.inception_input = self.max_pool_1
			with tf.variable_scope('1x1_branch'):
				with tf.variable_scope('initial_1x1'):
					self.initial_1x1 = layers.complete_conv2d(self.inception_input,currentDepth=32,newDepth=8,patch_size=1) # (N,14,14,8)
				with tf.variable_scope('5x5'):
					self.conv_5x5 = layers.complete_conv2d(self.initial_1x1,currentDepth=8,newDepth=16,patch_size=5) # (N,14,14,16)
				with tf.variable_scope('3x3'):
					self.conv_3x3 = layers.complete_conv2d(self.initial_1x1,currentDepth=8,newDepth=16,patch_size=3) # (N,14,14,16)
			with tf.variable_scope('avg_pool_branch'):
				with tf.variable_scope('initial_avg_pool'):
					self.initial_avg_pool = layers.average_pool(self.inception_input, stride=1) # (N,14,14,32)
				with tf.variable_scope('1x1'):
					self.conv_1x1 = layers.complete_conv2d(self.initial_avg_pool,currentDepth=32,newDepth=24,patch_size=1) # (N,14,14,24)
			with tf.variable_scope('concatenation'):
				self.inception_output = tf.concat(3, [self.initial_1x1, self.conv_5x5, self.conv_3x3, self.conv_1x1]) # (N,14,14,64)
		
		with tf.variable_scope('max_pool_2'):
			self.max_pool_2 = layers.max_pool(self.inception_output) # (N,7,7,64)
			
		image_size_after_conv = image_size/4
			
		self.reshaped_conv_output = tf.reshape(self.max_pool_2, [-1, image_size_after_conv*image_size_after_conv*64]) # (N,7*7*64)
		self.hidden = layers.simple_relu_layer(self.reshaped_conv_output, shape=[image_size_after_conv*image_size_after_conv*64,1024],dropout_keep_prob=self.dropout_keep_prob)
		return layers.simple_linear_layer(self.hidden, shape=[1024,num_labels])
		
class InceptionConvNetV2(AbstractModel):
	def __init__(self):
		super(InceptionConvNetV2,self).__init__()
		self.initial_learning_rate = 1e-3
		self.dropout_keep_prob = 0.5
		self.construct()
		
	def model(self,input):
		self.reshaped_input = tf.reshape(input, [-1,image_size,image_size,1]) # (N,28,28,1)
		
		with tf.variable_scope('conv_1'):
			self.conv_1 = layers.complete_conv2d(self.reshaped_input,currentDepth=1,newDepth=32,patch_size=5) # (N,28,28,32)
		with tf.variable_scope('max_pool_1'):
			self.max_pool_1 = layers.max_pool(self.conv_1) # (N,14,14,32)
		""" Inception module """
		with tf.variable_scope('inception_1'):
			self.inception_input = self.max_pool_1
			with tf.variable_scope('1x1_branch'):
				with tf.variable_scope('initial_1x1'):
					self.initial_1x1 = layers.complete_conv2d(self.inception_input,currentDepth=32,newDepth=8,patch_size=1) # (N,14,14,8)
				with tf.variable_scope('5x5'):
					self.conv_5x5 = layers.complete_conv2d(self.inception_input,currentDepth=32,newDepth=8,patch_size=1) # (N,14,14,8)
					self.conv_5x5 = layers.complete_conv2d(self.conv_5x5,currentDepth=8,newDepth=16,patch_size=5) # (N,14,14,16)
				with tf.variable_scope('3x3'):
					self.conv_3x3 = layers.complete_conv2d(self.inception_input,currentDepth=32,newDepth=8,patch_size=1) # (N,14,14,8)
					self.conv_3x3 = layers.complete_conv2d(self.conv_3x3,currentDepth=8,newDepth=16,patch_size=3) # (N,14,14,16)
			with tf.variable_scope('avg_pool_branch'):
				with tf.variable_scope('initial_avg_pool'):
					self.initial_avg_pool = layers.average_pool(self.inception_input, stride=1) # (N,14,14,32)
				with tf.variable_scope('1x1'):
					self.conv_1x1 = layers.complete_conv2d(self.initial_avg_pool,currentDepth=32,newDepth=24,patch_size=1) # (N,14,14,24)
			with tf.variable_scope('concatenation'):
				self.inception_output = tf.concat(3, [self.initial_1x1, self.conv_5x5, self.conv_3x3, self.conv_1x1]) # (N,14,14,64)
		
		with tf.variable_scope('max_pool_2'):
			self.max_pool_2 = layers.max_pool(self.inception_output) # (N,7,7,64)
			
		image_size_after_conv = image_size/4
			
		self.reshaped_conv_output = tf.reshape(self.max_pool_2, [-1, image_size_after_conv*image_size_after_conv*64]) # (N,7*7*64)
		self.hidden = layers.simple_relu_layer(self.reshaped_conv_output, shape=[image_size_after_conv*image_size_after_conv*64,1024],dropout_keep_prob=self.dropout_keep_prob)
		return layers.simple_linear_layer(self.hidden, shape=[1024,num_labels])