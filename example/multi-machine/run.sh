#!/bin/bash

nworker=$1
shift
nserver=$1
shift
config=$1
shift

set -x

../../dmlc-core/tracker/dmlc_mpi.py \
    -H hosts -n $nworker -s $nserver \
    ../../bin/cxxnet.ps $config update_on_server=1 param_server=dist $@

# ../../ps-lite/guide/local.sh $nserver $nworker \
#     ../../bin/cxxnet.ps $config update_on_server=1 param_server=dist -local
