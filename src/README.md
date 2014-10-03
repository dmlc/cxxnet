Coding Guide
======
This file is intended to be notes about code structure in cxxnet

Project Logical Layout
=======
* Dependency order: nnet->updater->layer
  - All module depends on global.h and utils
  - io is an independent module
* layer is implementation of neural net layers and defines forward and backward propagation
* updater is the parameter updating module, it defines update rule of weights
* nnet is the neural net structure that combines layers together to form a neural net
* io is the input module to handle reading various data and preprocessing

File Naming Convention
======= 
* .h files are data structures and interface, which are needed to use functions in that layer.
* -inl.hpp files are implementations of interface, like cpp file in most project.
  - You only need to understand the interface file to understand the usage of that layer
* In each folder, there can be a .cpp file, and .cu file that that compiles the module of that layer
  - the .cpp file and .cu file does not contain implementation, but reuse common implementation in file ends with _impl-inl.hpp
