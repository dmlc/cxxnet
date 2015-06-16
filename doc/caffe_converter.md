#### Introduction
This page will introduce the usage of Caffe converter. It can convert a pretrained model in Caffe to cxxnet format.

#### Preparation
* To begin with the convert, a latest version of Caffe should be built. 
* Currently, no automatic configuration file converter is provided. You need to convert the Caffe config file in prototxt to cxxnet configuration format by yourself. **Please make sure that all the layers in original Caffe model has corresponding layer in cxxnet!**
* Converters are provided in both C++ and Python.
 - To use the C++ converter, you need to specify the following paths in the config.mk. For example,

 ```bash
 # whether to build caffe converter
 USE_CAFFE_CONVERTER = 1
 CAFFE_ROOT = ~/caffe
 CAFFE_INCLUDE = ~/caffe/include/
 CAFFE_LIB = ~/caffe/build/lib/
 ```

 Then, run ```make all``` in the root of cxxnet. if everything is correct, you could find ```caffe_converter``` and ```caffe_mean_converter``` in the ```bin``` folder.

 - To use the Python converter, you should first make sure the Python wrapper of Caffe is successfully built. Then you need to specify the paths of Caffe and cxxnet in ```tools/caffe_converter/convert.py```.

#### Convert
* Simply run '''bin/caffe_converter''' and '''tools/caffe_converter/convert.py''', and then follow the instructions.
* To convert the mean file of Caffe, please use the C++ converter: ```caffe_mean_converter```. But we strongly recommend you to recompute the mean file in cxxnet due to the different data augmentation methods in Caffe and cxxnet.

