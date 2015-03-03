Python Interface example
====

CXXNET provides convenient interface for python user. We are able use Numpy array as training data source to train your network by using CXXNET in backend, we also are able extract feature into numpy array,

### Outline
* Import CXXNET
* Train a network by using ```train``` function
* Update the network
* Save and Load model
* Make prediction and evaluation
* Extract feature
* Advanced usage: ```Net``` object
* Advanced usage: ```Iterator``` object

Example script for MNIST can be found at [mnist.py](../example/MNIST/mnist.py)


### Import CXXNET
To import CXXNET into python, you need to add wrapper folder path to system path, eg
```python
CXXNET_WRAPPER_PATH = "/home/cxxnet/wrapper/"
import sys
sys.path.append(CXXNET_WRAPPER_PATH)
import cxxnet

```
### Train a network
CXXNET provides a simple method ```train``` to train a network. By following these steps, we can train a network in python.

#### Declare Network
Network structure is declared by configuration string, then we use the string to generate ```Net``` object for later task. The configuration format is same to original cxxnet, which can be found [here](layer.md)

For example, here is a simple network for MNIST.
```python
cfg = """
netconfig=start
layer[+1:fc1] = fullc:fc1
  nhidden = 100
  init_sigma = 0.01
layer[+1:sg1] = sigmoid:se1
layer[sg1->fc2] = fullc:fc2
  nhidden = 10
  init_sigma = 0.01
layer[+0] = softmax
netconfig=end

input_shape = 1,1,784
batch_size = 100

random_type = gaussian
"""
```
#### Declare Data
We can use both ```Iterator``` object and numpy ndarray as the training/evaluation data. We will discuss the ```Iterator``` in advanced feature, here we can use numpy ndarray directly.

```python
dtrain = data[0:1000, :] # training data
deval = data[1000:1500, :] # validation data
ltrain = label[0:1000, :]  # label for train
leval = label[1000:1500, :] #label for validation
```
For mini-batch data in numpy, we need some knowledge of ```Net``` object and ```update``` function in ```Net``` object so we will discuss it later.

#### Set Parameters
Parameters can be set in dictionary or list of tupple.
For example:
```python
param = {}
param['eta'] = 0.1
param['dev'] = 'cpu'
param['momentum'] = 0.9
param['metric'] = 'error'
```

#### Train this network, get the ```Net``` object
To train a net, there is a simple function ```cxxnet.train```. in our example, we can get a ```Net``` object in this way:
```python
num_round = 10
net = cxxnet.train(cfg=cfg, data=dtrain, label=ltrain, num_round, param)
```
Then in backend, cxxnet will be called and return the ```Net``` object for later advaced use.

### Update the network
We can use new data to update existing network. the data can be ```Iterator``` object or Numpy ndarray.
For example, when we have a ```Net``` object, we can use update to do mini-batch training like this:
```python
batch_size = 64
batch_num = data.shape[0] / batch_size
for i in xrange(batch_num):
  j = min((i + 1) * batch_size, data.shape[0])
  dbatch = data[i * batch_size : j, :]
  lbatch = label[i * batch_size : j, :]
  net.update(data=dbatch, label=lbatch)

```
To get ```Net``` object directly, we will discuss it in later advaced usage.
### Save and Load the network
We can save current network weights by calling
```python
file_name = "current.model"
net.save_model(file_name)
```
We can load existing model weight by calling
```python
file_name = "current.model"
net.load_model(file_name)
```
Please make sure both the ```Net``` object and existing model are in same network structure.

### Make Prediction and Evaluation
We can make prediction form net by using ```Iterator``` or Numpy ndarray. The prediction is stored in a numpy ndarray,
```python
dtest = data[1500:2000,:]
pred = net.predict(dtest)
```
For ```Iterator```, there is special function for evalutation in advanced usage. For numpy data, we can write an evaluation function to get the evaltion with prediction and label, eg
```python
werr = np.sum(label[:,0] !=  pred[:])
print "Error: %f" % werr
```
### Extract feature
To extract feature, we need both data and the node name which we will do extraction. A special node name format is ```top[-x]```, eg.
```python
raw_probability = net.extract(data, "top[-1]")
feature = net.extract(data, "fc7")
```
### Advanced usage: ```Net``` object
```Net``` object is able to be built by using ````train``` function, or we can initialize a network with configuration then train it by ourselves.

To get an ```Net``` object, we can do in this way
```python
cfg = """
netconfig=start
layer[+1:fc1] = fullc:fc1
  nhidden = 100
  init_sigma = 0.01
layer[+1:sg1] = sigmoid:se1
layer[sg1->fc2] = fullc:fc2
  nhidden = 10
  init_sigma = 0.01
layer[+0] = softmax
netconfig=end

input_shape = 1,1,784
batch_size = 100

random_type = gaussian
"""

net = cxxnet.Net(dev="gpu", cfg=cfg)
net.init_model()
```
Then we can use this net object just like previous net object to do update/predict/load/save/extract task.
We can also get/set weigth/bias for special layer in net.
eg:
```python
# get weight
weights = []
for layer in ['fc1', 'fc2']:
    for tag in ['wmat', 'bias']:
        weights.append((layer, tag, net.get_weight(layer, tag)))

# set weight
for layer, tag, wt in weights:
    net.set_weight(wt, layer, tag)
```
### Advanced usage: ```Iterator``` object
For large training task, for example, ImageNet training, we suggest to use CXXNET original iterator instead of training by numpy array directly because iterator is designed and implemented for best performance. To get an object, ```Iterator``` is very similar to ```Net```.
```python
data = cxxnet.DataIter("""
iter = mnist
    path_img = "./data/train-images-idx3-ubyte.gz"
    path_label = "./data/train-labels-idx1-ubyte.gz"
    shuffle = 1
iter = end
input_shape = 1,1,784
batch_size = 100
""")
print 'init train iter'

deval = cxxnet.DataIter("""
iter = mnist
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end
input_shape = 1,1,784
batch_size = 100
""")
```
There is a special ```train``` function for input is iterator, it is like:
```
net = train(cfg, data, num_round, param, eval_data = deval)
```
For update/predict/extract, you can use iterator object directly just like use numpy ndarray. The ```Net``` will get current batch of iterator then do the task. For evaluation, there is a special function ```evaluate``` for iterator. The ```evaluate``` function only accept iterator as input and automatically evaluate all batches in the iterator.

```python
# first parameter is iterator
# second parameter is name for this evaluation
print net.evaluate(deval, "test")
```

To go through all data by batch by using iterator, there are 3 useful functions:
```python
#reset the iterator to beginning
deval.before_first()

# get next batch, return true if success, false for iterator reaches end
deval.next()

# check whether it is at head/tail of current iterator
deval.check_valid()

```
To get current batch data or label to a numpy ndarray, we can use:
```
dbatch = deval.get_data()
lbatch = deval.get_label()
```
Here is an example to go through all batches in iterator and update the network
```python
data.before_first()
while data.next():
  net.update(data)

```
