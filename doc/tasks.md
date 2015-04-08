#### Introduction
This page will introduce the tasks supported in cxxnet, including:

* [Train](#train)
* [Predict](#predict)
* [Extract Features](#extract-features)
* [Finetune](#finetune)
* [Get Weight](#get-weight)

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
* To extract feature, you need to set task to ```extract```with node name or distance to top. ```model_in``` is also required to specify the model to use. The output of this task consists of two files: The first one is the extracted features. The second one is a meta-info file, which contrains the shape of the extracted features.
```bash
task = extract
extract_node_name = 45
model_in = ./models/0014.model
output_format = txt
```
```bash
task = extract
extract_node_name = top[-1]
model_in = ./models/0014.model
output_format = bin
# this will extract last node, namely the softmax prediction.
```
* **output_format** (default=txt) can be set to either "txt" or "bin". If setting to txt, then the output file is a text file seperated by space; otherwise the output file is a raw binary file. You may read the raw binary file using ```fread``` in C/C++ directly.
* **extract_node_name** indicates the node whose features should be extracted. For convenient, a special name ```top``` is used for extract toppest layer behind loss layer.


#### Finetune
To use finetune, you need to set ```task=finetune``` and ```model_in``` parameters in your global setting. Other parts are the same as task train. Note that finetune task will copy the parameters in the old network to the new one in the case that their layer names are exactly same. All other parts are initialized randomly. Note that ***You cannot copy a layer without a name.*** So it is a best practice that you add name for each layer, though it is not a must.

#### Get Weight
This task is used to get the learned weight from a layer. you need to set ```task=get_weight``` and ```model_in``` and ```weight_filename``` parameters in your global setting. The output format is the same as that in ```extract features``` task. Example:
```bash
task = get_weight
extract_layer_name = conv1
model_in = ./models/0014.model
weight_filename = weight.txt
```
this will extract the weights learned in ```conv1``` layer.


