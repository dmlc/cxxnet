#-----------------------------------------------------
#  cxxnet: the configuration compile script
#
#  This is the default configuration setup for cxxnet
#  If you want to change configuration, do the following steps:
#
#  - copy this file to the root folder
#  - modify the configuration you want
#  - type make or make -j n for parallel build
#----------------------------------------------------

# choice of compiler
export CC = gcc
export CXX = g++
export NVCC = nvcc

# whether use CUDA during compile
USE_CUDA = 1

# add the path to CUDA libary to link and compile flag
# if you have already add them to enviroment variable, leave it as NONE
USE_CUDA_PATH = NONE

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1

# whether use CUDNN library
USE_CUDNN = 0
# add the path to CUDNN libary to link and compile flag
# if you do not need that, or do not have that, leave it as NONE
USE_CUDNN_PATH = NONE

#
# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
USE_BLAS = mkl

# whether compile with parameter server
USE_DIST_PS = 0
PS_PATH = NONE
PS_THIRD_PATH = NONE

# the additional link flags you want to add
ADD_LDFLAGS = NONE

# the additional compile flags you want to add
ADD_CFLAGS = NONE
