# set LD_LIBRARY_PATH

export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../mshadow
export LDFLAGS= -lm -lcudart -lcublas -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lcurand
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX)

# specify tensor path
BIN = testc
OBJ = 
CUOBJ = cxxnet.o 
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

testc: testcompile.cpp cxxnet.o
cxxnet.o: cxxnet/cxxnet.cu cxxnet/*.hpp cxxnet/*.h

$(BIN) :
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
