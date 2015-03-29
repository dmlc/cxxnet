Coding Guide
======
This file is intended to be notes about code structure in cxxnet.
* The project follows Google's C code style
  - All the module interface are heavily documented in doxygen format
* Contribution is to the code and this NOTE welcomed!
* If you have questions on code, fire an github issue
  - If you want to help improve this note, send a pullreq

Getting Started
======
* In each folder in the src, you can find a ```.h``` file with the same name as the folder
  - These are interface of that module, heavily documented with doxygen comment
  - Start with these the interface header to understand the interface
* All the rest of the ```-inl.hpp``` files are implementations of the interface
  - These are invisible to other modules
  - Templatized class with parameter ```xpu``` that can stands for cpu or gpu
* The project depends on [mshadow](http://github.com/dmlc/mshadow) for tensor operations
  - You can find the documentation on mshadow in its repo.

Project Logical Layout
=======
* Dependency order: nnet->updater->layer
  - All module depends on global.h and utils
  - io is an independent module
* layer is implementation of neural net layers and defines forward and backward propagation
* updater is the parameter updating module, it defines update rule of weights
  - AsyncUpdater is a special updater that handles asynchronize communication and update
  - It uses [mshadow-ps](http://github.com/dmlc/mshadow/guide/mshadow-ps) to do async communication
* nnet is the neural net structure that combines layers together to form a neural net
   - Dependency in nnet: CXXNetThreadTrainer->NeuralNetThread->NeuralNet
* io is the input module to handle reading various data and preprocessing
  - io uses iterator pattern to handle data processing pipeline
  - The pipeline can be mult-threaded using threadbuffer trick

How do They Work Together
======
* Data is pulled from io module to feed into nnet
* nnet contains #gpu threads, that get part of data, call layer objects to do forwardbackprop
* For each weights, an updater is created
  - AsyncUpdater.AfterBackprop is called after backprop of the corresponding layer to push out gradient
  - AsyncUPdater.UpdateWait is called before forward to the layer
  - mshadow-ps does the async trick of parameter communication
* AsyncUpdater will call IUpdater, which does the updating trick
  - If update_on_server is on, IUpdater will be created on server-side instead

File Naming Convention
======= 
* .h files are data structures and interface
  - In each folder, there is one .h file that have same name as the folder, this file defines everything needed for other module to use this module
  - Interface headers: layer/layer.h, updater/updater.h
* -inl.hpp files are implementations of interface, like cpp file in most project.
  - You only need to understand the interface file to understand the usage of that layer
* In each folder, there can be a .cpp file, and .cu file that that compiles the module of that layer
  - the .cpp file and .cu file does not contain implementation, but reuse common implementation in file ends with _impl-inl.hpp
