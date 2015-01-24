MNIST Tutorial
=====

Reference Wiki: [https://github.com/antinucleon/cxxnet/wiki/tutorial](https://github.com/antinucleon/cxxnet/wiki/tutorial)


This tutorial contails basic configuration of feed forward netural network and convolution neural network on MNIST data set.

For detailed explanation please refer the wiki.

With distributed parameter server:


1. Build parameter server
```bash
git clone https://github.com/mli/parameter_server
cd parameter_server
./script/install_third.sh
make -j8
```

2. Build cxxnet with parameter server
```bash
git clone https://github.com/antinucleon/cxxnet -b V2-refactor
cd cxxnet
git clone https://github.com/tqchen/mshadow.git -b refactor
cd ..
cp make/config.mk .
sed -i.bak "s/USE_DIST_PS.*/USE_DIST_PS = 1/g" config.mk
make -j8
```

You can also place cxxnet at any other place, then you need to change `PS_PATH`
in `config.mk`

3. Run with 1 worker and 2 servers:

```bash
../../../script/local.sh ../../bin/cxxnet.ps 2 1 MNIST.conf -app_file MNIST.conf
```

TODO
1. it quite stuped to specify MNIST.conf twice
2. support multiple workers: each worker get a part of data, and print nicely
