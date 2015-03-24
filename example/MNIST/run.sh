#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data
fi

cd data

if [ ! -f "train-images-idx3-ubyte" ]; then
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gzip -d train-images-idx3-ubyte.gz
fi

if [ ! -f "train-labels-idx1-ubyte" ]; then
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gzip -d train-labels-idx1-ubyte.gz
fi

if [ ! -f "t10k-images-idx3-ubyte" ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gzip -d t10k-images-idx3-ubyte.gz
fi

if [ ! -f "t10k-labels-idx1-ubyte" ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gzip -d t10k-labels-idx1-ubyte.gz
fi

cd ..

if [ ! -d "models" ]; then
    mkdir models
fi


../../bin/cxxnet $1
