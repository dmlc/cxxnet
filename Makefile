# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
# all tge possible warning tread
export WARNFLAGS= -Wall -Wno-unused-parameter -Wno-unknown-pragmas
export IFLAGS=-I/opt/intel/mkl/include -I/usr/local/cuda-6.0/include/ -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/usr/local/cuda-6.0/lib64 
export CFLAGS = -g -O3 -msse3 -funroll-loops -I./mshadow/  -fopenmp -DMSHADOW_FORCE_STREAM $(IFLAGS) $(WARNFLAGS) 
export blas=0
export noopencv=0
export usecaffe=0

ifeq ($(blas),1)
 LDFLAGS= -lm -lcudart -lcublas -lcurand -lz -lcblas
 CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1
else
 LDFLAGS= -lm -lcudart -lcublas -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lcurand -lz
endif

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

export NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)

# specify tensor path
BIN = bin/cxxnet
OBJ = layer_cpu.o updater_cpu.o nnet_cpu.o data.o main.o
CUOBJ = layer_gpu.o  updater_gpu.o nnet_gpu.o
CUBIN =
.PHONY: clean all

ifeq ($(ps),1)
	CFLAGS += -DMSHADOW_DIST_PS_=1
	BIN += bin/cxxnet.ps
else
	CFLAGS += -DMSHADOW_DIST_PS_=0	
endif


all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

layer_cpu.o layer_gpu.o: src/layer/layer_impl.cpp src/layer/layer_impl.cu src/layer/*.h src/layer/*.hpp src/utils/*.h src/plugin/*.hpp
updater_cpu.o updater_gpu.o: src/updater/updater_impl.cpp src/updater/updater_impl.cu src/layer/layer.h src/updater/*.hpp src/updater/*.h src/utils/*.h
nnet_cpu.o nnet_gpu.o: src/nnet/nnet_impl.cpp src/nnet/nnet_impl.cu src/layer/layer.h src/updater/updater.h src/utils/*.h src/nnet/*.hpp src/nnet/*.h
data.o: src/io/data.cpp src/io/*.hpp
main.o: src/cxxnet_main.cpp 

bin/cxxnet: src/local_main.cpp $(OBJ) $(CUOBJ)
bin/cxxnet.ps: $(OBJ) $(CUOBJ) libps.a libps_main.a

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
