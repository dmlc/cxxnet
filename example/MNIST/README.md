This tutorial will show you how to use cxxnet train a feed forward neural network, then show you how to modify the feed forward network to a simple convolution neural network.

### Feed Forward Example
===
Reference configuration example [MNIST.conf]
(https://github.com/antinucleon/cxxnet/blob/master/example/MNIST/MNIST.conf)

### Before you start
* Run the command `./run.sh MNIST.conf` in the folder `example/MNIST/` to get data. 

### Setup data iterator configuration 
cxxnet use iterator to provide data batch to the network trainer. First we need to set the data type (_eval_ or _data_). Then we need to specify iterator type (_mnist_, _cifar_, _image_, _imgbin_, etc). Then set some attribute to the iterator including shuffle, file path and so on. 
Here is an example for MNIST
#### Setup training iterator
This part is about [**Data Iterator**](https://github.com/antinucleon/cxxnet/wiki/Data-Input) setting

* Change _**path_image**_ to the path of training image file, change _**path_label**_  to the path of training label file you download just now
```bash
data = train
iter = mnist
    path_img = "./data/train-images-idx3-ubyte.gz"
    path_label = "./data/train-labels-idx1-ubyte.gz"
    shuffle = 1
iter = end
```
#### Setup test iterator
* Change _**path_image**_ to the path of test image file, change _**path_label**_ to the path of test label file you download just now
```bash
eval = test
iter = mnist
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end
```
#### Setup network structure
* This part is about [**Layer Setting**](https://github.com/antinucleon/cxxnet/wiki/Layers)

* Network structure start with declaration _"netconfig=start"_ and end with _"netconfig=end"_. Note that there are two types of entities in cxxnet: node and layer. **Node** stores the intermediate results in the network, while **layer** denotes different kinds of transformations. Though the names of node and layer can be same, but they refer to different entities. They layer is declared in the format _"**layer[** from_node_name **->** to_node_name **]** = **layer_type**:name"_ Then comes the parameters of the layer. 
Here is an example for MNIST
```bash
netconfig=start
layer[0->1] = fullc:fc1
  nhidden = 100
  init_sigma = 0.01
layer[1->2] = sigmoid:se1
layer[2->3] = fullc:fc1
  nhidden = 10
  init_sigma = 0.01
layer[3->3] = softmax
netconfig=end
```
Notice some special layer like _softmax_ and _dropout_ use self-loop, which means the input node equals output node.
#### Setup input size and batch size
In this section, we need to set the input shape and batch size. The input shape should be 3 numbers, split by ','; The 3 numbers is channel, height and width for 3D input or 1, 1, dim for 1D vector input. In this example it is 
```bash
input_shape = 1,1,784
batch_size = 100
```
#### Setup global parameters
This part is about [**Global Setting**](https://github.com/antinucleon/cxxnet/wiki/Global-Setting)

Global parameters are used for setting the trainer behavior. In this example, we use the following configuration. 
```bash
dev = cpu
save_model = 15
max_round = 15
num_round = 15
train_eval = 1
random_type = gaussian
init_sigma = 0.01
```
First set working device **dev** ( _cpu_ or _gpu_ ); frequent to save mode **save_model** ; training round **num_round** and **max_round** and whether to print training set evaluation **train_eval**. The **random_type** defines weight initialization method. We provides ( _gaussian_ method and _xavier_ method)

#### Setup learning parameters
This part is about [**Updater Setting**](https://github.com/antinucleon/cxxnet/wiki/Updater)

learning parameter change the updater behavior. _eta_ is known as learning rate, and _wd_ is known as weight decay. And _momentum_ will help train faster.
```bash
eta = 0.1
momentum = 0.9
wd  = 0.0
```
Alternatively, we can set parameters specifically for bias and weight connections, using following configs.
```bash
wmat:eta = 0.1
bias:eta = 0.2
momentum = 0.9
wd  = 0.0
```
Here we set ```0.1``` learning rate for weight connections, and ```0.2``` learning rate for bias. Adding prefix ```wmat:``` to any parameter settings and they will be only applied to weight connections. While ```bias:```  stands for update parameters of bias 

#### Metric method
For classification, we use _error_ to metric performance. So set
```bash
metric = error
```
We also provide _logloss_, _rec@n_, _rmse_.
#### Running experiment
Make a folder named _"models"_ for saving the training model in the same folder you calling cxxnet_learner  
Then just run
```bash
../../bin/cxxnet ./MNIST.conf
```
or
```bash
./run.sh ./MNIST.conf
```
Then you will get a nearly 98% correct result in just several seconds.
#### Do prediction
Add these info in the configure file _MNIST.conf_
```bash
# Data iterator setting
pred = pred.txt
iter = mnist
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end
# Global Setting
task = pred
model_in = ./models/0014.model
```

run
```bash
../../bin/cxxnet MNIST.conf
```
The prediction result will be stored in the pred.txt. Other useful tasks include predict_raw (output the probability yields by the last classifier) and extract_feature (extract a given node specified in extract_layer_name). Please refer to Imagenet example for more details 

### Continue Training
===

Now we get a model file `models/0014.model`. 
Add these configuration into the original configuration file:
```bash
max_round = 3
model_in = ./models/0014.model
```
Then run `../../bin/cxxnet MNIST.conf`, it will continue train extra 3 round based on previous model.
If use `continue=1` instead of `model_in`, cxxnet will search the model folder and use the model file with largest number.

### Convolution Example
===
Reference configuration example [MNIST_CONV.conf](https://github.com/antinucleon/cxxnet/blob/master/example/MNIST/MNIST_CONV.conf)

Use convolution layer is easy in cxxnet. Based on previous configuration, make the following changes
* Data Iterator, set `input_flat = 0` to make input in 3D: (1,28,28)
```bash
# Data iterator setting
data = train
iter = mnist
    path_img = "./data/train-images-idx3-ubyte.gz"
    path_label = "./data/train-labels-idx1-ubyte.gz"
    input_flat = 0
    shuffle = 1
iter = end

eval = test
iter = mnist
    input_flat = 0
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end

```
* Update a convolution network structure
```bash
# Network structure setting
netconfig=start
layer[0->1] = conv:cv1
  kernel_size = 3
  pad = 1
  stride = 2
  nchannel = 32
  random_type = xavier
layer[1->2] = max_pooling
  kernel_size = 3
  stride = 2
layer[2->3] = flatten
layer[3->3] = dropout
  threshold = 0.5
layer[3->4] = fullc:fc1
  nhidden = 100
  init_sigma = 0.01
layer[4->5] = sigmoid:se1
layer[5->6] = fullc:fc1
  nhidden = 10
  init_sigma = 0.01
layer[6->6] = softmax
netconfig=end
```

* Set trainer input shape
```bash
input_shape = 1,28,28
```
* Try to use GPU train the conv net, set `dev=gpu`
```bash
dev = gpu
```
* Then run
```bash
../../bin/cxxnet ./MNIST_CONV.conf
```
or 
```bash
./run.sh ./MNIST_CONV.conf
```
You will get a result near 99% in a few seconds.
