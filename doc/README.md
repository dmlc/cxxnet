#Document Home

This is the documentation for cxxnet, a 

### Configuration file structure

A common structure for network configuration file is:

```
train_data_iterator_settings

validation_iterator_settings

netconfig=start
layer_settings
...
layer_settings
netconfig=end

updater_settings
global_setting

```

Sample configuration files can be found at example folder
* [Simple MNIST network](../example/MNIST/)
* [Kaggle Data Science Bowl Net](../example/kaggle_bowl)
* [AlexNet for ImageNet](../example/ImageNet)

### Parameter Setting
To understand more about cxxnet setting, this section will introduce all parameters of cxxnet and the effect of parameters. In general, cxxnet configuration file contains 3 kinds of configurations in a single file:

* [Data Input Iterator Setting](io.md)
  - Set input data configurations.

* [Layer Setting](layer.md)
  - Configure network, and setup each layers.

* [Updater Setting](updater.md)
  - Set parameters(learning rate, momentum) for learning procedure

* [Global Setting](global.md)
  - Set global parameters for neural network, related to device selection, running control.
