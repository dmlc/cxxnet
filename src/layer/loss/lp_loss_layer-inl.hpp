#ifndef CXXNET_LAYER_LP_LOSS_LAYER_INL_HPP_
#define CXXNET_LAYER_LP_LOSS_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "./loss_layer_base-inl.hpp"

namespace cxxnet {
namespace layer {
/*! \brief loss function layer */
template<typename xpu>
class LpLossLayer: public LossLayerBase<xpu> {
 public:
  LpLossLayer(const LabelInfo *label_info)
      : LossLayerBase<xpu>(label_info) {
    p = 2.0;
  }
  virtual ~LpLossLayer(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "p")) p = atof(val);
    LossLayerBase<xpu>::SetParam(name, val);
  }
 protected:
  virtual void Forward_(mshadow::Tensor<xpu, 2> inout_data,
                        mshadow::Stream<xpu> *stream) {
    // Do Nothing
  }
  virtual void SetGradCPU(mshadow::Tensor<cpu, 2> inout_data,
                          const LabelRecord &label) {
    mshadow::Tensor<cpu, 2> lb = label.label;
    utils::Assert(lb.size(0) == inout_data.size(0) && lb.size(1) == inout_data.size(1),
                  "LpLossLayer: label size mismatch");
    for (index_t i = 0; i < inout_data.size(0); ++i) {
      for (index_t j = 0; j < inout_data.size(1); ++j) {
        inout_data[i][j] = p * std::pow(std::abs(inout_data[i][j] - lb[i][j]), p - 1)
          * (inout_data[i][j] > lb[i][j] ? 1 : -1);
      }
    }
  }
 private:
  // L_p loss
  float p;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_LP_LOSS_LAYER_INL_HPP_
