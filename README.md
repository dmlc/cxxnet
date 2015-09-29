# We move forward to [MXNet](https://github.com/dmlc/mxnet) !
----

Dear users,

Thanks for using and supporting cxxnet. Today, we finally make a hard but exciting decision: **we decide to deprecate cxxnet and fully move forward to next generation toolkit [MXNet](https://github.com/dmlc/mxnet).**

Please check the feature [highlights](https://github.com/dmlc/mxnet#features), [speed/memory comparation](https://github.com/dmlc/mxnet/tree/master/example/imagenet) and [examples](https://github.com/dmlc/mxnet/tree/master/example) in MXNet.


cxxnet developers,

28th, Sep, 2015


-----
Note: We provide a very simple converter to MXNet. Check [guide](simple_mxnet_converter) to see whether your model is able to be converted. 

------

#cxxnet


CXXNET is a fast, concise, distributed deep learning framework.

Contributors: https://github.com/antinucleon/cxxnet/graphs/contributors

* [Documentation](doc)
* [Learning to use cxxnet by examples](example)
* [Note on Code](src)
* User Group(TODO)

###Feature Highlights

* Lightweight: small but sharp knife
  - cxxnet contains concise implementation of state-of-art deep learning models
  - The project maintains a minimum dependency that makes it portable and easy to build
* Scale beyond single GPU and single machine
  - The library works on multiple GPUs, with nearly linear speedup
  - THe library works distributedly backed by disrtibuted parameter server
* Easy extensibility with no requirement on GPU programming
  - cxxnet is build on [mshadow](#backbone-library)
  - developer can write numpy-style template expressions to extend the library only once
  - mshadow will generate high performance CUDA and CPU code for users
  - It brings concise and readable code, with performance matching hand crafted kernels
* Convenient interface for other languages
  - Python interface for training from numpy array, and prediction/extraction to numpy array
  - Matlab interface

### News
* 24-May, 2015: Pretrained [Inception model](example/ImageNet/Inception-BN.conf) with 89.9% Top-5 Correctness is ready to use.
* 09-Apr, 2015: Matlab Interface is ready to use


### Backbone Library
CXXNET is built on [MShadow: Lightweight CPU/GPU Tensor Template Library](https://github.com/tqchen/mshadow)
* MShadow is an efficient, device invariant and simple tensor library
  - MShadow allows user to write expressions for machine learning while still provides
  - This means developer do not need to have knowledge on CUDA kernels to extend cxxnet.
* MShadow also provides a parameter interface for Multi-GPU and distributed deep learning
  - Improvements to cxxnet can naturally run on Multiple GPUs and being distributed

###Build

* Copy ```make/config.mk``` to root foler of the project
* Modify the config to adjust your enviroment settings
* Type ```./build.sh``` to build cxxnet
