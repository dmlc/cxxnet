#### Introduction
This page will introduce some advanced usages in cxxnet, including:
* [Multi-label Training](#multi-label-training)

#### Multi-label Training
* To use multi-label training, you need the following three steps in additional to the case of single label training:
	- For multi-label training, in ```imgrec```, you need to specify ```image_list``` field to indicate the list file that contains the labels.
	- First, you need to specify the number of labels in the network by setting ```label_width``` variable in global settings. The following setting denotes that we have 5 labels in the network.
	```bash
	label_width = 5
	```
	- In the [image list file](io.md#image-list-file), you need to provide ```label_width``` labels instead of one label. Namely, each line is in the format:
	```
	image_index \t label_1 \t label_2 ... \t label_n \t file_name
	```
	- In global setting, you need to specify how each field of the labels form a label vector. For example, we are interested in a localization task. In the task, we first need to output the label for one image, and next predict its position denoted by a bounding box. The configuration can be written as:
	```
	label_vec[0,1) = class
	label_vec[1,5) = bounding_box
	```
	- At last, in each loss layer, you need to specify the target of the loss:
	```
	layer[19->21] = softmax
		target = class
	layer[20->22] = l2_loss
		target = bounding_box
	```
	This means for the first field of the labels, we treat it as a class label, and apply standard softmax loss function on it. For the other four labels, we treat them as the coordinates of the bounding box, and train them using Euclidean loss.
	