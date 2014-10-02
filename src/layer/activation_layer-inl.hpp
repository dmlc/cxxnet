#ifndef CXXNET_LAYER_ACTIVATION_LAYER_INL_HPP_
#define CXXNET_LAYER_ACTIVATION_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu,typename ForwardOp, typename BackOp>
class ActivationLayer : public CommonLayerBase<xpu>{
 public:
  ActivationLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
  virtual ~ActivationLayer(void) {}

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    pnode_out->data.shape = node_in.data.shape;
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    pnode_in->data = F<ForwardOp>(pnode_in->data);
    mshadow::Copy(pnode_out->data, pnode_in->data);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (prop_grad) {
      pnode_in->data = F<BackOp>(pnode_in->data) * pnode_out->data;
    }
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_ACTIVATION_LAYER_INL_HPP_

