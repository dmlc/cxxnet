export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../mshadow
export LDFLAGS= -lcblas -lm -lz

BIN = cxxnet_learner.cpu
CFLAGS += -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CUDA=0

BIN = cxxnet_learner.cpu
OBJ = cxxnet_data.o  cxxnet_nnet_cpu.o 
.PHONY: clean all

all: $(BIN) $(OBJ) 

cxxnet_nnet_cpu.o: ../cxxnet/nnet/cxxnet_nnet.cpp ../cxxnet/core/*.hpp ../cxxnet/core/*.h ../cxxnet/nnet/*.hpp ../cxxnet/nnet/*.h 
cxxnet_data.o: ../cxxnet/io/cxxnet_data.cpp ../cxxnet/io/*.hpp ../cxxnet/utils/cxxnet_io_utils.h
cxxnet_learner.cpu: ../cxxnet/cxxnet_main.cpp cxxnet_data.o cxxnet_nnet_cpu.o

$(BIN) :
	$(CXX) $(CFLAGS) $(filter %.cpp %.o %.c, $^) -o $@ $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

clean:
	$(RM) $(OBJ) $(BIN) *~
