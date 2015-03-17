#ifndef CXXNET_LAYER_MULTISIGMOID_LAYER_INL_HPP_
#define CXXNET_LAYER_MULTISIGMOID_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "./loss_layer_base-inl.hpp"

namespace cxxnet {
namespace layer {
/*! \brief loss function layer */
template<typename xpu>
class MultiLogisticLayer: public LossLayerBase<xpu> {
 public:
  MultiLogisticLayer(const LabelInfo *label_info)
      : LossLayerBase<xpu>(label_info) {}
  virtual ~MultiLogisticLayer(void) {
  }
 protected:
  virtual void Forward_(mshadow::Tensor<xpu, 2> inout_data,
                        mshadow::Stream<xpu> *stream) {
    inout_data = mshadow::expr::F<op::sigmoid>(inout_data);
  }
  virtual void SetGradCPU(mshadow::Tensor<cpu, 2> inout_data,
                          const LabelRecord &label) {
    mshadow::Tensor<cpu, 2> lb = label.label;
    utils::Assert(lb.size(0) == inout_data.size(0) && lb.size(1) == inout_data.size(1),
                  "MultiLogisticLayer: label size mismatch");
    for (index_t i = 0; i < inout_data.size(0); ++i) {
      for (index_t j = 0; j < inout_data.size(1); ++j) {
        inout_data[i][j] -= lb[i][j];
      }
    }
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_MULTISIGMOID_LAYER_INL_HPP_
