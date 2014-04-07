cxxnet
======

CXXNET is a neural network toolkit build on mshadow(https://github.com/tqchen/mshadow).


Creater: [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/) and [Bing Xu](http://ca.linkedin.com/in/binghsu)

## Introduction
cxxnet is yet another implementation of (convolutional) neural network. It is in C++, about 1000 lines of [layer implementations](../blob/master/cxxnet/core/cxxnet_layer-inl.hpp), easily configuration via config file, and can get the state of art performance.

## Features
* Light amount of code: cxxnet is implemented in C++. It is powered by [mshadow](https://github.com/tqchen/mshadow), and powerful light weight matrix and tensor template. All the functions are only implemented once. As a result, the core part of the implementation is less than 2000 lines, and easily extendible.
* Speed:  On Bing Xu’s EVGA GeForce 780 GTX with 2304 CUDA cores, cxxnet archived 211 images per second in training on ImageNet data with Alex Krizhevsky’s deep network structure. It means one round of ImageNet training can be done in less 2 hours, and is able to train more than  18 million images per day.


## License
cxxnet is licenced in Apache License, Version 2.0 (refer to the [LICENSE](https://github.com/antinucleon/cxxnet/blob/master/LICENSE) for details)


