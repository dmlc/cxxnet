cxxnet
======

CXXNET is a neural network toolkit build on mshadow(https://github.com/tqchen/mshadow).


Creater: [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/) and [Bing Xu](http://ca.linkedin.com/in/binghsu)

## Introduction
cxxnet is yet another implementation of (convolutional) neural network. It is in C++, about 1000 lines of [layer implementations](../blob/master/cxxnet/core/cxxnet_layer-inl.hpp), easily configuration via config file, and can get the state of art performance.

## Features
* Small but sharp knife: cxxnet is implemented in C++.  the core part of the implementation is less than 2000 lines, and easily extendible.
  -This is due to the use of [mshadow](https://github.com/tqchen/mshadow), a powerful light weight matrix and tensor template. All the functions are only implemented once. As a result

* Speed:  On Bing Xu’s EVGA GeForce 780 GTX with 2304 CUDA cores, cxxnet archived 211 images per second in training on ImageNet data with Alex Krizhevsky’s deep network structure. It means one round of ImageNet training can be done in less 2 hours, and is able to train more than  18 million images per day.


