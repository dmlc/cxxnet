#### Introduction 
This page introduce updater setting for cxxnet

* [General SGD Updater](https://github.com/antinucleon/cxxnet/wiki/Updater#updater)
* [Constant Learning Rate Scheduling](https://github.com/antinucleon/cxxnet/wiki/Updater#constant-scheduling)
* [Exp Decay Learning Rate Scheduling](https://github.com/antinucleon/cxxnet/wiki/Updater#exp-decay)
* [Poly Decay Learning Rate Scheduling](https://github.com/antinucleon/cxxnet/wiki/Updater#poly-decay)
* [Factor Decay Learning Rate Scheduling](https://github.com/antinucleon/cxxnet/wiki/Updater#factor-decay)

#### Updater
In default, the cxxnet will use the SGDUpdater.
The `eta`, `wd` and `momentum` can be set differently to `wmat` or `bias` (namely, the weight and bias in each layer), by configure 
```bias
wmat:eta = 0.1
bias:eta = 0.2
```
If not specify the target, the setting will take effect globally.  
* Basic configuration:
```bash
updater = sgd
eta = 0.01
momentum = 0.9
wd = 0.0005
```
* **eta** is known as learning rate, default is 0.01 
* **momentum** is momentum, default is 0.9
* **wd** is known as weight decay (l2 regularization), default is 0.005

* Global updater setting can be **overridden** in the layer configuration. eg.
```bash
# Global setting
eta = 0.01
momentum = 0.9
# Layer setting
netconfig=start
layer[0->1] = fullc:fc1
  nhidden = 100
  eta = 0.02
  momentum = 0.5
layer[1->2] = sigmoid:se1
layer[2->3] = fullc:fc1
  nhidden = 10
layer[3->3] = softmax
netconfig=end
```
In the layer `fc1`, the learning rate will be `0.02` and momentum will be `0.5`, but layer `fc2` will follow the global setting, whose learning rate will be `0.01` and momentum will be `0.9`


=
#### Learning Rate Scheduling
There are some advanced features for SGDUpdater, like learning rate scheduling. We provides four learning rate scheduling method: _constant_ , _expdecay_ and _polydecay_ and _factor_.

#### Common parameters
* **lr:start_epoch** start learning rate scheduling after iteration, default is 0. Before that, we use constant learning rate specified in _eta_.
* **lr:minimum_lr** minimum of learning rate, default is 0.0001
* **lr:step** the parameter used in each scheduling method, elaborate in each method

#### Constant Scheduling
* Example of **constant** scheduling: In this way the learning rate keep same
```bash
updater = sgd
eta = 0.01
momentum = 0.9
wd = 0.0005
lr:schedule = constant
```

#### Exp Decay
Exponential Learning rate decay adjust learning rate like this formula:
`new_learning_rate = base_learning_rate * pow(gamma, iteration / step )`
* Example of **expdecay** scheduling: In this way the learning rate drop in exponential way
```bash
updater = sgd
eta = 0.01
momentum = 0.9
wd = 0.0005
lr:schedule = expdecay
lr:start_iteration = 3000
lr:minimum_lr = 0.001
lr:gamma = 0.5
lr:step = 1000
```
* **lr:gamma** learning decay param, default is 0.5

#### Poly Decay
Polynomial learning rate decay adjusts the learning rate like this formula:
`new_learning_rate = base_learning_rate * pow( 1.0 + (iteration/step) * gamma, -alpha )` 
* Example of **polydecay** scheduling: In this way the learning rate drop in polynomial way
```bash
updater = sgd
eta = 0.01
momentum = 0.9
wd = 0.0005
lr:schedule = polydecay
lr:start_epoch = 3000
lr:minimum_lr = 0.001
lr:alpha = 0.5
lr:gamma = 0.1
lr:step = 1000
```
* **lr:gamma** learning decay param, default is 0.5
* **lr:alpha** learning decay param, default is 0.5

#### Factor Decay
Factor Decay multiplies the learning rate by _factor_ each _step_ iterations. It is also called step scheduling.
* Example of **factor** scheduling:
```bash
updater = sgd
eta = 0.01
momentum = 0.9
wd = 0.0005
lr:schedule = factor
lr:minimum_lr = 0.001
lr:factor = 0.1
lr:step = 10000
```
* **lr:factor** learning decay param, default is 0.1
