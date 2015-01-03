#ifndef CXXNET_LAYER_FLATTEN_LAYER_INL_HPP_
#define CXXNET_LAYER_FLATTEN_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class FlattenLayer : public ILayer<xpu> {
 public:
  virtual ~FlattenLayer(void) {}
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "FlattenLayer: only support 1-1 connection");
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    nodes_out[0]->data.shape_ = 
        mshadow::Shape4(ishape[0], 1, 1, ishape[1] * ishape[2] * ishape[3]);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;    
    nodes_out[0]->data = reshape(nodes_in[0]->data, nodes_out[0]->data.shape_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (prop_grad) {
      nodes_in[0]->data = reshape(nodes_out[0]->data, nodes_in[0]->data.shape_);
    }    
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_FLATTEN_LAYER_INL_HPP_

