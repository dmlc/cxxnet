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
        struct sigmoid {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 1.0f / (1.0f + expf(-a));
            }
        };
        struct sigmoid_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a * ( 1.0f - a );
            }
        };

        /*! \brief Rectified Linear Operation
                   https://en.wikipedia.org/wiki/Rectifier_(neural_networks) */
        struct relu {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return max( a, 0.0f );
            }
        };
        struct relu_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a > 0.0f ? 1.0f : 0.0f;
            }
        };

        struct tanh {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return tanhf( a );
            }
        };
        struct tanh_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 1.0f - a * a;
            }
        };

    }; //namespace op

}; //namespace cxxnet

#endif // CXXNET_OP_H
