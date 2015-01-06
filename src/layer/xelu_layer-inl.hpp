#ifndef LAYER_XELU_LAYER_INL_HPP_
#define LAYER_XELU_LAYER_INL_HPP_
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template <typename xpu>
class XeluLayer : public ILayer<xpu> {
 public:
  virtual ~XeluLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "ActivationLayer Layer only support 1-1 connection");
    nodes_out[0]->data.shape_ = nodes_in[0]->data.shape_;
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    // InitConnection is already called, no need to check size again
    nodes_in[0]->data = F<op::xelu>(nodes_in[0]->data, param_.xelu);
    mshadow::Copy(nodes_out[0]->data, nodes_in[0]->data);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (prop_grad) {
      nodes_in[0]->data = F<op::xelu_grad>(nodes_in[0]->data, param_.xelu) * nodes_out[0]->data;
    }
  }
 private:
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
};

} // namespace layer
} // namespace xelu

#endif // XELU_LAYER_INL_HPP_
