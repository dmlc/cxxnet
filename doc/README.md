#Document Home

This is the documentation for cxxnet

### Starter Guide

Before you start, we highly recommend you to go through the [MNIST example](../example/MNIST/). including detailed tutorial of how to make a configuration file from zero, then train a fully connected network to a [convolution network] then to MPI distributed example. There is also an example for training MNIST network with Python interface. There are also more [examples](../example) for you to have a better view of how to use cxxnet.



### Parameter Setting
To understand more about cxxnet setting, this section will introduce all parameters of cxxnet and the effect of parameters. In general, cxxnet configuration file contains 4 kinds of configurations in a single file:

* [Data Input Iterator Setting](io.md)
  - Set input data configurations.

* [Layer Setting](layer.md)
  - Configure network, and setup each layers.

* [Updater Setting](updater.md)
  - Set parameters(learning rate, momentum) for learning procedure

* [Global Setting](global.md)
  - Set global parameters for neural network, related to device selection, running control.
