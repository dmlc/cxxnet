ImageNet Tutorial
=====

This is a step by step example of AlexNet for ImageNet Challenge.

### Introduction
This tutorial will guide you train your own super vision model. The default configure will take more than 3GB GPU RAM, so make batch size smaller or larger according to your GPU RAM size.

* Normally, smaller batch_size means more noise in gradient, and maybe a smaller learning rate is needed.
* If you want to still use a large batch with not enough RAM, You can set ```update_period=2``` and ```batch_size=128```, this means the parameter update is done every 2 batches, which is equivalent to ```batch_size=256```

### 0.Before you start
Make sure you have downloaded the ImageNet training data. Resize the picture into size 256 * 256 *3 for later we will crop random 227 * 227 * 3 image while training.

### 1.Make the image list
After you get the data, you need to make a [image list file](../../doc/io.md#image-list-file) first.  The format is
```
integer_image_index \t label_index \t path_to_image
```
In general, the program will take a list of names of all image, shuffle them, then separate them into training files name list and testing file name list. Write down the list in the format.

A sample file is provided here
```bash
895099  464     n04467665_17283.JPEG
10025081        412     ILSVRC2010_val_00025082.JPEG
74181   789     n01915811_2739.JPEG
10035553        859     ILSVRC2010_val_00035554.JPEG
10048727        929     ILSVRC2010_val_00048728.JPEG
94028   924     n01980166_4956.JPEG
1080682 650     n11807979_571.JPEG
972457  633     n07723039_1627.JPEG
7534    11      n01630670_4486.JPEG
1191261 249     n12407079_5106.JPEG

```

### 2.Make the binary file
Although you can use image iterator now. the disk random seek will make the training process extremely slow. So **you'd better generate binary file for training and use imgbin iterator** .

To generate binary image, you need to use *im2bin* in the tool folder. The im2bin will take the path of _image list file_ you generated just now, _root path_ of the images and the _output file path_ as input. These processes usually take several hours, so be patient. :)

A sample command:
```bash
im2bin ./train.lst ./resized256_images/ TRAIN.BIN
```
### 3.Set correct configuration file
Change the iterator path in the [ImageNet.conf](ImageNet.conf) to point to your _image list file_ and _image binary file_ correctly, then just run as MNIST example. After about 20 round, you can see some reasonable result.
By calling
```bash
cxxnet ./ImageNet.conf 2>eval.log
```
You can save the evaluation result into the file `eval.log`



=
### Acknowledgment
* Reference: Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." NIPS. Vol. 1. No. 2. 2012.
* The network parameter scheduling is adapted from configuration provided by [Caffe](http://caffe.berkeleyvision.org/)
