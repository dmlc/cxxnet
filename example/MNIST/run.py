#! /usr/bin/python
import os
import sys
import subprocess

urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", \
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", \
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", \
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
if os.path.exists("./data") == False:
    os.mkdir("./data")
    os.chdir("./data")
    for url in urls:
        p = subprocess.Popen(["wget", url])
        p.wait()
    os.chdir("..")
else:
    os.chdir("./data")
    for url in urls:
        name = url.split("/")[-1]
        if os.path.exists(name) == False:
            p = subprocess.Popen(["wget", url])
            p.wait()
    os.chdir("..")

if os.path.exists("./models") == False:
    os.mkdir("./models")


if len(sys.argv) != 2:
    print "Usage: python run.py Configure"
    sys.exit(-1)

if os.path.exists(sys.argv[1]) == False:
    print "Configuration %s not exist!" % sys.argv[1]
    sys.exit(-1)

p = subprocess.Popen(["../../cxxnet_learner", sys.argv[1]])
p.wait()



