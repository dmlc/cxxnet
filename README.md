cxxnet
======

CXXNET (spelled as: C plus plus net) is a neural network toolkit build on mshadow(https://github.com/tqchen/mshadow). It is yet another implementation of (convolutional) neural network. It is easily configured via config file, and can get the state of art performance.


People: [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/), [Naiyan Wang](http://winsty.net/), [Mu Li](https://www.cs.cmu.edu/~muli/), Bing Xu


[Documentation and Tutorial](doc)

## Features
* Small but sharp knife: the core part of the implementation is less than 2000 lines
* Based on parameter-server, cxxnet supports multi-GPU training and distributed training with elegant speed.
* Build with [mshadow](https://github.com/tqchen/mshadow), a tensor template library for unified CPU/GPU computation. All the functions are only implemented once, as a result. cxxnet is easy to be extended by writing tensor expressions.
* Python/Matlab interface for training and prediction. 


## Build Guide
cxxnet is designed to require less third party library. The minimal requirement is MKL/CBLAS/OpenBLAS and MShadow(which can be downloaded automatically). Other dependence can be set by editing  [make/config.mk](make/config.mk) before make.

* For users who want train neural network in less time, we suggest you buy a NVIDIA cuda-enabled video card and install CUDA in your system, then set ```USE_CUDA = 1``` in [make/config.mk](make/config.mk) to enable GPU training.
* For users who want to better speed up on convolution neural network, we suggest you install CuDNN R2 and set ```USE_CUDNN=1``` in [make/config.mk](make/config.mk).
* For users who want to train on images, libjpeg or libjpeg-turbo is required for decoding images. We suggest you install OpenCV and set ```USE_OPENCV=1``` to enable augmentation iterator.
* For MKL users who want to use Python interface, we suggest you change MShadow make config file to link to MKL in ```static``` way.
