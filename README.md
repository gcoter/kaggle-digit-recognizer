# Kaggle Digit Recognizer
The goal of this repository is to store the different models I tried so far for the Kaggle Digit Recognizer competition.

## Scripts
Here is the list of the scripts I wrote :

* constants.py : contains all constants.
* datasetmanagement.py : defines functions to manage the MNIST dataset.
* layers.py : defines functions to create some basic structures used in models with tensorflow.
* models.py : defines the different models. All models inherit "AbstractModel" which defines common elements (placeholders, loss, optimizer, etc...).
* train_ops.py : defines functions for training.
* train.py : I run this script for my tests. It calls functions from other scripts to construct the dataset, to define a model (easy to change), to train it and to generate the submission file.
* kaggle_script.py : I ran this script directly on Kaggle.

I used train.py to run tests (since it allows me to easily change the model I want to test) and then copied the useful code in kaggle_script.py (with one given model I chose).

## Models
Here is the list of the models I tested so far :

* "Simple" multi-layers perceptron (Simple MLP)
* Convolutional network
* Convolutional network with Inception module (2 versions)

I use Adam for optimization and cross entropy to calculate the loss.

### Simple MLP
I simply stacked fully connected layers.

I used RELU as activation function between hidden layers and Softmax for the output layer. 

To avoid overfitting, I also added dropout. I tried L2 regularization as well.

### Convolutional network
For the moment, I implemented a quite classical structure :

* First convolutional layer (5x5 patch, stride 1, depth : 32)
* First max pooling layer (2x2 kernel, strides 2)
* Second convolutional layer (5x5 patch, stride 1, depth : 64)
* Second max pooling layer (2x2 kernel, strides 2)
* A simple MLP

This structure shows great improvements, even if it is much longer to train, especially on my CPU (44 seconds for 100 steps).

### Inception Convolutional Network
I read a lot about the Inception module designed by Google (used in GoogLeNet) and decided to try it. I implemented two versions:

* Version 1 : the first 1x1 convolution shares its output with the 3x3 and the 5x5 convolutions
* Version 2 : there is one 1x1 convolution for each 3x3 or 5x5 convolutions

So I took the previous network and replaced the second convolution with an inception module.

I noticed it speeds up training on my CPU (24 seconds for 100 steps) while keeping a very good accuracy.

## Results

| Model                  | Epochs | Batch size | Real time   | Best accuracy achieved |
| ---------------------- |:------:|:----------:|:-----------:|:----------------------:|
| Simple MLP             | 10     | 100        | 3 min 41 s  | 94.70%                 |
| Simple ConvNet         | 10     | 50         | 59 min 48 s | 98.67%                 |
| Inception ConvNet V1   | 20     | 50         | 76 min 55 s | 98.51%                 |
| Inception ConvNet V2   | 10     | 100        | 33 min 11 s | 95.20%                 |

## References
Competition : https://www.kaggle.com/c/digit-recognizer

Inception : 

* GoogLeNet's paper: http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
* Udacity video describing one inception module: https://www.youtube.com/watch?v=VxhSouuSZDY