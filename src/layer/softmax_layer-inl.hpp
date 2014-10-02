#ifndef CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_
#define CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_

#include "./layer.h"
#include "mshadow/tensor.h"

namespace cxxnet {
namespace layer {
/*! \brief softmax layer, do softmax transformation during forward */
template<typename xpu>
class SoftmaxLayer: public CommonLayerBase<xpu>{
 public:
  SoftmaxLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {
    utils::Check(p_in == p_out, "softmax layer must self loop e.g layer[1->1] = softmax");
  }
  virtual ~SoftmaxLayer(void) {}

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in, Node<xpu> *pnode_out) {}
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    mshadow::Softmax(pnode_out->mat(), pnode_out->mat() );
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_SOFTMAX_LAYER_INL_HPP_
