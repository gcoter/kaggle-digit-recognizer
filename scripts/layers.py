import tensorflow as tf

def new_weights(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def new_biases(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# === LAYERS ===
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