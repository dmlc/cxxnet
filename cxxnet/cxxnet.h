#ifndef CXXNET_H
#define CXXNET_H
#pragma once
/*!
 * \file cxxnet.h
 * \brief Base definition for cxxnet
 * \author Bing Xu
 */

#include "mshadow/tensor.h"
#include "cxxnet_node.h"

namespace cxxnet {

typedef mshadow::cpu cpu;
typedef mshadow::gpu gpu;

};


#endif // CXXNET_H
