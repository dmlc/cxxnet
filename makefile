# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
# all tge possible warning tread
export WARNFLAGS= -Wall -Wno-unused-parameter -Wno-unknown-pragmas
export IFLAGS=-I/opt/intel/mkl/include -I/usr/local/cuda-6.0/include/ -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/usr/local/cuda-6.0/lib64
export CFLAGS = -g -O3 -msse3 -funroll-loops -I./mshadow/  -fopenmp -DMSHADOW_FORCE_STREAM $(IFLAGS) $(WARNFLAGS)
export blas=0
export noopencv=1
export usecaffe=0
export usecudnn=0
# ifeq ($(blas),1)
#  LDFLAGS= -lm -lcudart -lcublas -lcurand -lz -lcblas
#  CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1
# else
CFLAGS+=-DMSHADOW_USE_CUDA=0 -std=c++0x -I../../src -I../../third_party/include -DMSHADOW_DIST_PS_=1
PS_LIB=../../build/libps.a ../../build/psmain.o
LDFLAGS+= -L../third_party/lib -lgflags -lzmq -lprotobuf -lglog -lsnappy
LDFLAGS+= -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lz
# endif

ifeq ($(noopencv),1)
	CFLAGS+= -DCXXNET_USE_OPENCV=0
else
	CFLAGS+= -DCXXNET_USE_OPENCV=1
	LDFLAGS+= `pkg-config --libs opencv`
endif

ifeq ($(usecaffe), 1)
	LDFLAGS+= -L./caffe/ -lcaffe -lglog -lprotobuf
	CFLAGS+= -DCXXNET_USE_CAFFE_ADAPTOR=1 -I./caffe/include -I/home/winsty/glog-0.3.3/src -I/home/winsty/leveldb-master/include
else
	CFLAGS+= -DCXXNET_USE_CAFFE_ADAPTOR=0
endif

ifeq ($(usecudnn), 1)
	CFLAGS+= -DCXXNET_USE_CUDNN=1 -I/home/winsty/cudnn
	LDFLAGS+= -L/home/winsty/cudnn -lcudnn
endif
export NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)

# specify tensor path
BIN = bin/cxxnet.ps
OBJ = layer_cpu.o updater_cpu.o nnet_cpu.o data.o main.o nnet_ps_server.o
CUOBJ = layer_gpu.o  updater_gpu.o nnet_gpu.o
CUBIN =
.PHONY: clean all


ifeq ($(ps),1)
	CFLAGS += -DMSHADOW_DIST_PS_=1
	BIN += bin/cxxnet.ps
else
	CFLAGS += -DMSHADOW_DIST_PS_=0
endif


all: $(BIN)

layer_cpu.o layer_gpu.o: src/layer/layer_impl.cpp src/layer/layer_impl.cu src/layer/*.h src/layer/*.hpp src/utils/*.h src/plugin/*.hpp
updater_cpu.o updater_gpu.o: src/updater/updater_impl.cpp src/updater/updater_impl.cu src/layer/layer.h src/updater/*.hpp src/updater/*.h src/utils/*.h
nnet_cpu.o nnet_gpu.o: src/nnet/nnet_impl.cpp src/nnet/nnet_impl.cu src/layer/layer.h src/updater/updater.h src/utils/*.h src/nnet/*.hpp src/nnet/*.h
nnet_ps_server.o: src/nnet/nnet_ps_server.cpp src/utils/*.h src/nnet/*.hpp src/nnet/*.h
data.o: src/io/data.cpp src/io/*.hpp
main.o: src/cxxnet_main.cpp

bin/cxxnet: src/local_main.cpp $(OBJ) #$(CUOBJ)
bin/cxxnet.ps: src/dist_ps_server.cpp $(OBJ) $(PS_LIB)

$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c %.a, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)
$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~ */*~ */*/*~
