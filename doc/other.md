#### Introduction
This page will introduce other setting in cxxnet, including:
* [Device Selection](#set-working-hardware)
* [Printing Control](#print-information)
* [Training Round](#set-round-of-training)
* [Saving Model and Continue Training](#saving-model-and-continue-training)
* [Prediction](#prediction)
* [Extract Feature](#extract-feature)
* [Fine-tune](#fine-tune)



#### Set working hardware
* To use CPU, set the field
```bash
dev = cpu
```
* To use GPU, set the field
```bash
dev = gpu
```
We can also set specific device (say device 1) by using
```bash
dev = gpu:1
```
* To use multi-GPU, set the field with the corresponding device id
```bash
dev = gpu:0,1,2,3
```
In default, it is `dev=gpu`


#### Print information
* To print training error evaluation, just set this field to 1
```bash
eval_train = 1
```
* in default this field is 0, which means cxxnet won't print anything about training error.
* To turn off all information while training, set this field to 1
```bash
silent = 1
```
* In default this field is 0
* To control print frequent, change this field
```bash
print_step = 100
```
* In default it will print every 100 batch


#### Set round of training
There are two field handle training round together: _**num_round**_ and _**max_round**_
* _**num_round**_ is used for number of round to train
* _**max_round**_ is used for maximum number of round to train from now on
```bash
num_round = 15
max_round = 15
```
This configuration will make cxxnet train for 15 rounds on the training data.

More examples,
```bash
num_round = 50
max_round = 2
```
If we have a model trained 40 rounds, then use this configuration continue to train, cxxnet will stop at the 42 round.


#### Saving model and continue training
* To save model while training round, set this field to saving frequent(a number)
```bash
save_model = 2
model_dir = path_of_dir_to_save_model
```
* In default, this field is 1, means cxxnet will save a model in every round
* To continue a training process, you need to set model_in as the input snapshot you want to continue from
```conf
model_in = path of model file
```
* Alternatively, if you save model every round (save_model=1), then you can use option continue, cxxnet will automatically search the latest model and start from that model
```conf
continue = 1
```
In default, if neither of the two values is set, cxxnet will start training from start.


#### Prediction
* In default, cxxnet treats the configuration file as a training configuration. To make it predict, you need to add extra data iterator and specify the task to be `pred` and model you want to use to do prediction. For example
```bash
# Data Iterator Setting
pred = pred.txt
iter = mnist
iterator_optition_1 = ..
iterator_optition_1 = ...
iter = end
# Global Setting
task = pred
model_in = ./models/0014.model
```
* In which the _*mode_in*_ is the path to the model which we need to use for prediction. The _*pred*_ field is the file we will save the result. The iterator configuration is same to traditional iterator.
* To get the softmax prediction directly, set the task to
```bash
task = extract
extract_node_name = top[-1]
```
#### Extract Feature
* To extract feature, you need to set task to ```extract_feature```with node name or distance to top
```bash
task = extract_feature
extract_node_name = 45
```
```bash
task = extract_feature
extract_node_name = top[-2]
# this will extract top feature layer
```

If you didn't set name for layer, just use the number as name; If you set special name, use the name. For convenient, a special name ```top``` is used for extract topest layer behind loss layer.

#### Fine-tune
TODO
