# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -g -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I./mshadow/ -I/opt/intel/mkl/include -I/usr/local/cuda-6.0/include/ -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/usr/local/cuda-6.0/lib64 -fopenmp
export blas=0
export noopencv=1
ifeq ($(blas),1)
 LDFLAGS= -lm -lcudart -lcublas -lcurand -lz -lcblas
 CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1
else
 LDFLAGS= -lm -lcudart -lcublas -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lcurand -lz
endif
ifeq ($(xgboost),1)
	CFLAGS+= -DCXXNET_ADAPT_XGBOOST=1 -I../xgboost
endif

ifeq ($(noopencv),1)
	CFLAGS+= -DCXXNET_USE_OPENCV=0
else
	CFLAGS+= -DCXXNET_USE_OPENCV=1
	LDFLAGS+= `pkg-config --libs opencv`
endif

export NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)

# specify tensor path
BIN = bin/cxxnet
OBJ = layer_cpu.o updater_cpu.o nnet_cpu.o data.o
CUOBJ = layer_gpu.o  updater_gpu.o nnet_gpu.o
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

layer_cpu.o layer_gpu.o: src/layer/layer_impl.cpp src/layer/layer_impl.cu src/layer/*.h src/layer/*.hpp src/utils/*.h
updater_cpu.o updater_gpu.o: src/updater/updater_impl.cpp src/updater/updater_impl.cu src/layer/layer.h src/updater/*.hpp src/updater/*.h src/utils/*.h
nnet_cpu.o nnet_gpu.o: src/nnet/nnet_impl.cpp src/nnet/nnet_impl.cu src/layer/layer.h src/updater/updater.h src/utils/*.h src/nnet/*.hpp src/nnet/*.h 
data.o: src/io/data.cpp src/io/*.hpp

bin/cxxnet: src/cxxnet_main.cpp $(OBJ) $(CUOBJ)

$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)
$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~ */*~ */*/*~
