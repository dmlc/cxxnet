#ifndef CXXNET_H
#define CXXNET_H
#pragma once#ifndef CXXNET_H
#define CXXNET_H
#pragma once
/*!
 * \file cxxnet.h
 * \brief Base definition for cxxnet
 * \author Bing Xu
 */
#include "mshadow/tensor.h"

/*! \brief namespace of cxxnet */
namespace cxxnet {
    typedef mshadow::cpu cpu;
    typedef mshadow::gpu gpu;
    typedef mshadow::index_t index_t;
    typedef mshadow::real_t  real_t;
};

namespace cxxnet {

};

#include "cxxnet_net.h"

#endif // CXXNET_H
