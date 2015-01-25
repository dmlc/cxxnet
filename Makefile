ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

# use customized config file
include $(config)

# all tge possible warning tread
WARNFLAGS= -Wall -Wno-unused-parameter -Wno-unknown-pragmas
CFLAGS = -g -O3 -msse3 -funroll-loops -I./mshadow/  -fopenmp
CFLAGS += -DMSHADOW_FORCE_STREAM $(WARNFLAGS)
LDFLAGS = -lm -lz -pthread
NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)

ifeq ($(USE_CUDA), 0)
	CFLAGS += -DMSHADOW_USE_CUDA=0
else
	LDFLAGS += -lcudart -lcublas -lcurand
endif
ifneq ($(USE_CUDA_PATH), NONE)
	CFLAGS += -I$(USE_CUDA_PATH)/include
	LDFLAGS += -L$(USE_CUDA_PATH)/lib64
endif

ifeq ($(USE_BLAS), mkl)
ifneq ($(USE_INTEL_PATH), NONE)
	LDFLAGS += -L$(USE_INTEL_PATH)/mkl/lib/intel64
	LDFLAGS += -L$(USE_INTEL_PATH)/lib/intel64
	CFLAGS += -I$(USE_INTEL_PATH)/mkl/include
endif
	LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
else
	CFLAGS += -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
endif
ifeq ($(USE_BLAS), openblas)
	LDFLAGS += -lopenblas
else ifeq ($(USE_BLAS), atlas)
	LDFLAGS += -lcblas
else ifeq ($(USE_BLAS), blas)
	LDFLAGS += -lblas
endif


# setup opencv
ifeq ($(USE_OPENCV),1)
	CFLAGS+= -DCXXNET_USE_OPENCV=1
	LDFLAGS+= `pkg-config --libs opencv`
else
	CFLAGS+= -DCXXNET_USE_OPENCV=0
endif

# customize cudnn path
ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DCXXNET_USE_CUDNN=1
endif
ifneq ($(USE_CUDNN_PATH), NONE)
	CFLAGS += -I$(USE_CUDNN_PATH)
	LDFLAGS += -L$(USE_CUDNN_PATH) -lcudnn
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif
ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

ifeq ($(PS_PATH), NONE)
PS_PATH = ..
endif
ifeq ($(PS_THIRD_PATH), NONE)
PS_THIRD_PATH = $(PS_PATH)/third_party
endif

ifeq ($(USE_DIST_PS),1)
CFLAGS += -DMSHADOW_DIST_PS=1 -std=c++11 \
	-I$(PS_PATH)/src -I$(PS_THIRD_PATH)/include
PS_LIB = $(addprefix $(PS_PATH)/build/, libps.a libpsmain.a) \
	$(addprefix $(PS_THIRD_PATH)/lib/, libgflags.a libzmq.a libprotobuf.a \
	libglog.a libz.a libsnappy.a)
NVCCFLAGS += --std=c++11
else
	CFLAGS+= -DMSHADOW_DIST_PS=0
endif


# specify tensor path
BIN = bin/cxxnet
OBJ = layer_cpu.o updater_cpu.o nnet_cpu.o data.o main.o nnet_ps_server.o
CUOBJ = layer_gpu.o  updater_gpu.o nnet_gpu.o
CUBIN =
ifeq ($(USE_CUDA), 0)
	CUDEP =
else
	CUDEP = $(CUOBJ)
endif

.PHONY: clean all

ifeq ($(USE_DIST_PS), 1)
BIN=bin/cxxnet.ps
endif

all: $(BIN)

layer_cpu.o layer_gpu.o: src/layer/layer_impl.cpp src/layer/layer_impl.cu\
	src/layer/*.h src/layer/*.hpp src/utils/*.h src/plugin/*.hpp

updater_cpu.o updater_gpu.o: src/updater/updater_impl.cpp src/updater/updater_impl.cu\
	src/layer/layer.h src/updater/*.hpp src/updater/*.h src/utils/*.h

nnet_cpu.o nnet_gpu.o: src/nnet/nnet_impl.cpp src/nnet/nnet_impl.cu src/layer/layer.h\
	src/updater/updater.h src/utils/*.h src/nnet/*.hpp src/nnet/*.h

nnet_ps_server.o: src/nnet/nnet_ps_server.cpp src/utils/*.h src/nnet/*.hpp \
	src/nnet/*.h mshadow/mshadow-ps/*.h
data.o: src/io/data.cpp src/io/*.hpp

main.o: src/cxxnet_main.cpp

bin/cxxnet: src/local_main.cpp $(OBJ) $(CUDEP)
bin/cxxnet.ps: $(OBJ) $(CUDEP) $(PS_LIB)

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
