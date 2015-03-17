Document Home
====
This is the documentation for cxxnet

Links of Resources
* [Learning CXXNET by Examples](../example)
* [Python Interface](python.md)
* [Multi-GPU/Distributed Training](multigpu.md)

Configuration of CXXNET
====
This section introduces the how to setup configuation file of cxxnet.
In general, cxxnet configuration file contains four kinds of configurations in a single file:
* [Data Input Iterator Setting](io.md)
  - Set input data configurations.
* [Layer Setting](layer.md)
  - Configure network, and setup each layers.
* [Updater Setting](updater.md)
  - Set parameters(learning rate, momentum) for learning procedure
* [Tasks](tasks.md)
  - This page includes all the four tasks you could try by cxxnet.
* [Other Setting](other.md)
  - Set other parameters for neural network, related to device selection, running control.
