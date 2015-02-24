CXXNET Example
====

This folder contains all the code examples using cxxnet
* Contribution of examples, benchmarks is more than welcome!
* If you like to share how you use xgboost to solve your problem, send a pull request:)


[Feature Walk-through by using MNIST](MNIST)
====

This is basic sample configuration and usage demo by using MNIST, including detailed tutorial of how to make a configuration file from zero, then train a [fully connected network](MNIST/MNIST.conf) to a [convolution network](MNIST/MNIST_CONV.conf), then to [MPI distributed example](MNIST/mpi.conf). There is also an example for training MNIST network with [Python interface](MNIST/mnist.py).

[Kaggle National Data Science Bowl Example](kaggle_bowl)
====
This is an example to show you how to solve a real kaggle problem by using cxxnet. The model is a convolution neural network with run-time augmentation. This example also show basic steps to using convolution neural network:
1. resize image
2. make list file for train and test data
3. make validation dataset (optinal)
4. pack image to binary page file
5. make network configuration file
6. train model, adjust parameter
7. make a submission!

detailed instructions are in the [kaggle_bowl](kaggle_bowl) folder

[ImageNet Example](ImageNet)
====
This is a step by step example to show how to use cxxnet train an AlexNet for ImageNet task. We also provides better reference pre-trained model with all configurations.
