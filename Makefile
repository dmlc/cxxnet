ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = dmlc-core
endif

ifneq ($(USE_OPENMP_ITER),1)
	export NO_OPENMP = 1
endif

# use customized config file
include $(config)
include mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk
unexport NO_OPENMP

# all tge possible warning tread
WARNFLAGS= -Wall
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)
CFLAGS += -g -O3 -I./mshadow/ -I./dmlc-core/include -fPIC $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
ROOTDIR = $(CURDIR)

# setup opencv
ifeq ($(USE_OPENCV),1)
	CFLAGS+= -DCXXNET_USE_OPENCV=1
	LDFLAGS+= `pkg-config --libs opencv`
else
	CFLAGS+= -DCXXNET_USE_OPENCV=0
endif

ifeq ($(USE_OPENCV_DECODER),1)
	CFLAGS+= -DCXXNET_USE_OPENCV_DECODER=1
else
	CFLAGS+= -DCXXNET_USE_OPENCV_DECODER=0
endif

ifeq ($(USE_OPENMP_ITER), 1)
	CFLAGS += -fopenmp
endif

# customize cudnn path
ifneq ($(USE_CUDNN_PATH), NONE)
	CFLAGS += -I$(USE_CUDNN_PATH)
	LDFLAGS += -L$(USE_CUDNN_PATH)
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DCXXNET_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

# specify tensor path
BIN = bin/cxxnet 
ifeq ($(USE_OPENCV),1)
	BIN += bin/im2rec bin/bin2rec
endif
SLIB = wrapper/libcxxnetwrapper.so
OBJ = layer_cpu.o updater_cpu.o nnet_cpu.o main.o nnet_ps_server.o
OBJCXX11 = data.o
CUOBJ = layer_gpu.o  updater_gpu.o nnet_gpu.o
CUBIN =

LIB_DEP = $(DMLC_CORE)/libdmlc.a
ifeq ($(USE_RABIT_PS), 1)
	LIB_DEP += $(RABIT_PATH)/lib/librabit_base.a
endif

ifeq ($(USE_CUDA), 0)
	CUDEP =
else
	CUDEP = $(CUOBJ)
endif

.PHONY: clean all

ifeq ($(USE_DIST_PS), 1)
	BIN=bin/cxxnet.ps
endif

all: $(BIN) $(SLIB)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

$(RABIT_PATH)/lib/librabit_base.a:
	+ cd $(RABIT_PATH); make; cd $(ROOTDIR)

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

wrapper/libcxxnetwrapper.so: wrapper/cxxnet_wrapper.cpp $(OBJ) $(OBJCXX11) $(LIB_DEP) $(CUDEP)
bin/cxxnet: src/local_main.cpp $(OBJ) $(OBJCXX11) $(LIB_DEP) $(CUDEP)
bin/cxxnet.ps: $(OBJ) $(OBJCXX11) $(CUDEP) $(LIB_DEP) $(PS_LIB)
bin/im2rec: tools/im2rec.cc $(DMLC_CORE)/libdmlc.a
bin/bin2rec: tools/bin2rec.cc $(DMLC_CORE)/libdmlc.a

$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(OBJCXX11) :
	$(CXX) -std=c++0x -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(OBJCXX11) $(BIN) $(CUBIN) $(CUOBJ) $(SLIB) *~ */*~ */*/*~
	cd $(DMLC_CORE); make clean; cd -
