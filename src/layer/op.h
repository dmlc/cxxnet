#ifndef CXXNET_LAYER_OP_H_
#define CXXNET_LAYER_OP_H_
#pragma once
/*!
 * \file op.h
 * \brief extra mshadow operation for cxxnet
 * \author Bing Xu
 */
#include <mshadow/tensor.h>

namespace cxxnet {
/*! \brief operations for ActivationLayer */
namespace op {
/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
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

struct softplus {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return logf(1 + expf(a));
  }
};

struct softplus_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};

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
