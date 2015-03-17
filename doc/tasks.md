#### Introduction
This page will introduce the tasks supported in cxxnet, including:

* [Train](#train)
* [Predict](#predict)
* [Extract Features](#extract-features)
* [Finetune](#finetune)

####Train
* Train is the basic task for cxxnet. If you don't specify the task in global configuration, the task is train by default.
* To use ```train```, you must specify a data iterator to indicate the training data, which starts with ```data = train``` e.g.
```bash
data = train
iter = mnist
    iterator_optition_1 = ..
	iterator_optition_2 = ...
iter = end
```

You can also specify data iterators for evaluation if needed. The iterator should start with ```eval = iter_name``` e.g.
```bash
eval = test
iter = mnist
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end
```
More details about iterator, please refer to [**Data Iterator**](io.md).

#### Predict
* In default, cxxnet treats the configuration file as a training configuration. To make it predict, you need to add extra data iterator and specify the task to be `pred` and model you want to use to do prediction. For example
```bash
# Data Iterator Setting
pred = pred.txt
iter = mnist
	iterator_optition_1 = ..
	iterator_optition_2 = ...
iter = end
# Global Setting
task = pred
model_in = ./models/0014.model
```
* In which the _*mode_in*_ is the path to the model which we need to use for prediction. The _*pred*_ field is the file we will save the result. The iterator configuration is same to traditional iterator.

#### Extract Features
* To extract feature, you need to set task to ```extract```with node name or distance to top. ```model_in``` is also required to specify the model to use.
```bash
task = extract
extract_node_name = 45
model_in = ./models/0014.model
```
```bash
task = extract_feature
extract_node_name = top[-1]
model_in = ./models/0014.model
# this will extract last node, namely the softmax prediction.
```

For convenient, a special name ```top``` is used for extract topest layer behind loss layer.


#### Finetune
To use finetune, you need to set ```task=finetune``` and ```model_in``` parameters in your global setting. Other parts are the same as task train. Note that finetune task will copy the parameters in the old network to the new one in the case that their layer names are exactly same. All other parts are initialized randomly. Note that ***You cannot copy a layer without a name.*** So it is a best practice that you add name for each layer, though it is not a must.


