# Kaggle Digit Recognizer
The goal of this repository is to store the different models I tried so far for the Kaggle Digit Recognizer competition.

## Algorithms
Here is the list of the models I tested so far :

* "Simple" multi-layers perceptron (Simple MLP)
* Convolutional network

### Simple MLP
This model is the most simple :

* One input layer (784 neurons)
* Some hidden layers
* One output layer (10 neurons)

I used RELU as activation function between hidden layers and Softmax for the output layer. 

To avoid overfitting, I also added dropout. I tried L2 regularization as well.

### Convolutional network
For the moment, I implemented a quite classical structure :

* First convolutional layer (5x5 patch, stride 1, depth : 32)
* First max pooling layer (2x2 kernel, strides 2)
* Second convolutional layer (5x5 patch, stride 1, depth : 64)
* Second max pooling layer (2x2 kernel, strides 2)
* A simple MLP

## Results

| Model                 | Best accuracy achieved |
| --------------------- |:----------------------:|
| Simple MLP            | 85.6%                  |
| Convolutional network | 97.8%                  |


## References
https://www.kaggle.com/c/digit-recognizer