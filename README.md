cxxnet
======

CXXNET (spelled as: C plus plus net) is a neural network toolkit build on mshadow(https://github.com/tqchen/mshadow). It is yet another implementation of (convolutional) neural network. It is in C++, with about 1000 lines of [network layer implementations](https://github.com/antinucleon/cxxnet/blob/master/cxxnet/core/cxxnet_layer-inl.hpp), easily configuration via config file, and can get the state of art performance.


Creater: [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/) and [Bing Xu](http://ca.linkedin.com/in/binghsu)

Documentation and Tutorial: https://github.com/antinucleon/cxxnet/wiki

## Features
* Small but sharp knife: the core part of the implementation is less than 2000 lines, and easily extendible.
  - cxxnet is build with [mshadow](https://github.com/tqchen/mshadow), a tensor template library for unified CPU/GPU computation. All the functions are only implemented once, as a result.
* Speed:  On Bing Xu’s EVGA GeForce 780 GTX with 2304 CUDA cores, cxxnet archived 211 images per second in training on ImageNet data with Alex Krizhevsky’s deep network structure. The prediction speed is 400 pic / second on the same card.

## Build Guide
* Common Requirement:  NVIDIA CUDA with cuBLAS, cuRAND and cudaRT; OpenCV; mshadow (will be downloaded by using build.sh)
* MKL version: Intel MKL directly run `build.sh`
* If you don’t have MKL, using `build.sh blas=1` to build with CBLAS
    - Depending your version of CBLAS(ATLAS, etc.), you may need to change -lblas to -lcblas in Makefile 
