#ifndef LAYER_FLATTEN_LAYER_INL_HPP_
#define LAYER_FLATTEN_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class FlattenLayer : public CommonLayerBase<xpu> {
 public:
  FlattenLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
  virtual ~FlattenLayer(void) {}

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    mshadow::Shape<4> ishape = node_in.data.shape;
    pnode_out->data.shape = mshadow::Shape4(ishape[3], 1, 1, ishape[2] * ishape[1] * ishape[0]);
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;    
    pnode_out->data = reshape(pnode_in->data, pnode_out->data.shape);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (prop_grad) {
      pnode_in->data = reshape(pnode_out->data, pnode_in->data.shape);
    }
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_FLATTEN_LAYER_INL_HPP_

