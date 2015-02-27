Multi-GPU / Distributed Training
======
This page contains

[Set Multi-GPU in configuration file]()

[Make cxxnet work in distributed system]()

[How it works]()

[Reference]()

### Set Multi-GPU in configuration file
* To use multi-GPU, set the field with the corresponding device id
```bash
dev = gpu:0,1,2,3
```
which indicate cxxnet will use the first four GPU to do the training task

### Make cxxnet work in distributed system


### How it works
Parameter Server is the backend of multi-gpu / distributed training part of cxxnet. For multi-gpu, the parameter is running on local machine so you don't need to set mannually.

For distributed case TODO



### Reference
