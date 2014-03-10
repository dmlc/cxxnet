#ifndef CXXNET_OP_H
#define CXXNET_OP_H
#pragma once
/*!
 * \file cxxnet_op.h
 * \brief extra mshadow operation for cxxnet
 * \author Bing Xu
 */
#include "mshadow/tensor.h"

namespace cxxnet {
    /*! \brief operations for algorithm */
    namespace op {
        /*! \brief Rectified Linear Operation
                   https://en.wikipedia.org/wiki/Rectifier_(neural_networks) */
        struct relu {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a > 0 ? a : 0;
            }
        }; // struct relu

    }; //namespace op

}; //namespace cxxnet

#endif // CXXNET_OP_H
