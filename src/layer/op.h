#ifndef CXXNET_LAYER_OP_H_
#define CXXNET_LAYER_OP_H_
#pragma once
/*!
 * \file op.h
 * \brief extra mshadow operation for cxxnet
 * \author Bing Xu
 */
#include "mshadow/tensor.h"

namespace cxxnet {
namespace op{
struct square {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a;
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

/*! \brief used for generate element of power */
struct power {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return powf( a, b );
  }
};

}  // namespace op
}  // namespace cxxnet
#endif // CXXNET_LAYER_OP_H
