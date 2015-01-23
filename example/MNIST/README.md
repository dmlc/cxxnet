MNIST Tutorial
=====

Reference Wiki: [https://github.com/antinucleon/cxxnet/wiki/tutorial](https://github.com/antinucleon/cxxnet/wiki/tutorial)


This tutorial contails basic configuration of feed forward netural network and convolution neural network on MNIST data set.

For detailed explanation please refer the wiki.

Here is a temp hacker to run with paramter server:

Build:

first build paramater server https://github.com/mli/parameter_server
then put cxxnet into plugin/, and build it by `make -f mk.ps`
now you can run it by use 1 worker and 1 server:

> ../../../../script/local.sh ../../bin/cxxnet.ps 1 1 MNIST.conf

it should be ok to append any other augments to cxxnet
