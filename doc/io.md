#### Introduction
This page will introduce data input method in cxxnet. cxxnet use data iterator to provide data to the neural network.  Iterators do some preprocessing and generate batch for the neural network.

* We provide basic iterators for MNIST, Image, Binary Image and CSV.
* To boost performance, we provide thread buffer for loading.
  - Putting threadbuffer iterator after input iterator will open an independent thread to fetch from the input, this allows parallelism of learning process and data fetching.
  - We recommend you use thread buffer in all cases to avoid IO bottle neck.

Declarer the iterator in the form
```bash
iter = iterator_type
options 1 =
options 2 =
...
iter = end
```
* The basic iterator type is **mnist** , **image** , **imgbin**, **csv**
* To use thread buffer, declare in this form
```bash
iter = iterator_type
options 1 =
options 2 =
...
iter = threadbuffer
iter = end
```
=
**Iterators**
* [MNIST](#mnist-iterator)
* [Image and Image Binary](#image-and-image-binary-iterator)
* [CSV Iterator](#csv-iterator)


=
##### Preprocessing Options
```bash
shuffle = 1
```
* **shuffle** set 1 to shuffle the **training data**. Note that this option **does not** apply to  **imgbin** and **csv**.

=
### MNIST Iterator
* Required fields
```bash
path_img = path to gz file of image
path_label = path to gz file of label
input_flat = 1
```
* **input_flat** means loading the data in shape 1,1,784 or 1,28,28
* You may check a full example [here](https://github.com/antinucleon/cxxnet/blob/master/example/MNIST/MNIST.conf)

=
##### Image and Image Binary Iterator
There are two ways to load images, image iterator that takes list of images in the disk, and image binary iterator that reads images from a packed binary file. Usually, I/O is a bottle neck, and image binary iterator makes training faster. However, we also provide image iterator for convenience


### Image Iterator
* Required fields
```bash
image_list = path to the image list file
image_root = path to the image folder
```
###### Image list file
The **image_list** is a formatted file. The format is
```c++
image_index \t label \t file_name
```

A valid image list file is like the following (NO header):
```bash
1       0       cat.5396.jpg
2       0       cat.11780.jpg
3       1       dog.11254.jpg
4       0       cat.6791.jpg
5       0       cat.7937.jpg
6       1       dog.9329.jpg
```


* **image_root** is the path to the folder contains files in the image list file.

##### Image binary iterator
Image binary iterator aims to reduce to IO cost in random seek. It is especially useful when deal with large amount for data like in ImageNet.
* Required field
```bash
image_list = path to the image list file
image_bin = path to the image binary file
```
* The **image_list** file is described [above](#image-list-file)
* To generate **image_bin** file, you need to use the tool [im2bin](https://github.com/antinucleon/cxxnet/blob/master/tools/im2bin.cpp) in the tools folder.
* You may check an example [here](https://github.com/antinucleon/cxxnet/blob/master/example/ImageNet/ImageNet.conf)

#### Realtime Preprocessing Option for Image/Image Binary
```bash
rand_crop = 1
rand_mirror = 1
divideby = 256
image_mean = "img_mean.bin"
mean_value=255,255,255
min_crop_size=40
max_crop_size=80
max_aspect_ratio = 0.5
max_shear_ratio=0.3
max_rotate_angle=180
```

=
##### Common Parameters
* **divideby** normalize the data by dividing a value
* **image_mean** minus the image by the mean of all image. The value is the path of the mean image file. If the file doesn't exist, cxxnet will generate one.
* **mean_value** minus the image by the value specified in this field. Note that only one of **image_mean** and **mean_value** should be specified.

=
##### Random Augmenations
* **rand_crop** set 1 for randomly cropping image of size specified in **input_shape**. If set to 0, the iterator will only output the center crop.
* **rand_mirror** set 1 for random mirroring the **training data**
* **min_crop_size** and **max_crop_size** denotes the range of crop size. If they are not 0, the iterator will randomly pick _x_ in [min_crop_size, max_crop_size]. And then it will crop a region whose width and height are _x_. At last, the crop region is resize to **input_shape**.
* **max_aspect_ratio** denotes the max ratio of random aspect ratio augmentation. If it is not 0, the iterator will first random width in [min_crop_size, max_crop_size], and then random _aspect_ratio_ in [0, max_aspect_ratio]. The height is set to `y = max(min_crop_size, min(max_crop_size, x * (1 + aspect_ratio)))`. After cropping, the region is resized to **input_shape**.
* **max_shear_ratio** denotes the max random shearing ratio. In training, the image will be sheared randomly in [0, max_shear_ratio].
* **max_rotate_angle** denotes the random rotation angle. In training, the image will be rotated randomly in [-max_rotate_angle, max_rotate_angle].
* **rotate_list** specifies a list that input will rotate. e.g. `rotate_list=0,90,180,270` The input will only rotate randomly in the set.
* **max_random_contrast** denotes the range of random contrast variation. The output will be `y = (x - mean) * (1 + contrast)`, where `x` is the original image, and `contrast` is randomly picked in [-max_random_contrast, max_random_contrast]. **It will not take effect unless mean_value or mean_file specified.**
* **max_random_illumination** denotes the range of random illumination variation. The output will be `y = (x - mean) * contrast`, where `x` is the original image, and `illumination` is randomly picked in [-max_random_illumination, max_random_illumination]. **It will not take effect unless mean_value or mean_file specified.**

=
##### Deterministic Transformations
Deterministic transformations are usually used in test to generate diverse prediction results. Ensembling diverse prediction results could improve the performance.
* **crop_x_start** and **crop_y_start**  denotes the left corner of the crop.
* **mirror** denotes whether mirror the input.
* **rotate** denotes the angle will rotate.

=
### CSV Iterator
This iterator can be used to read data files that stores in a raw csv file. In each line, the first **label_width** fields (default = 1, which means the label of the sample is a scalar.) are the labels for the data samples, the rest are the features for each dimension. e.g.
```bash
iter = csv
filename = "train.csv"
has_header=0
```
* **filename** denotes the file name of the csv file.
* **has_header** denotes whether this csv has header line, if this parameter is set to 1, the iterator will automatically skip the first line.
