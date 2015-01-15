#ifndef CXXNET_LAYER_SPLIT_LAYER_INL_HPP_
#define CXXNET_LAYER_SPLIT_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"


namespace cxxnet {
namespace layer {

template<typename xpu>
class SplitLayer : public ILayer<xpu> {
 public:
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    //utils::Check(nodes_in.size() == 1 && nodes_out.size() > 1,
    //             "Split layer only support 1-n connection");
    mshadow::Shape<4> oshape = nodes_in[0]->data.shape_;
    for (index_t i = 0; i < nodes_out.size(); ++i){
      nodes_out[i]->data.shape_ = oshape;
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    for (index_t i = 0; i < nodes_out.size(); ++i){
      mshadow::Copy(nodes_out[i]->data, nodes_in[0]->data,
        nodes_out[i]->data.stream_);
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    if (prop_grad){
      mshadow::Copy(nodes_in[0]->data, nodes_out[0]->data,
        nodes_in[0]->data.stream_);
      for (index_t i = 1; i < nodes_out.size(); ++i){
        nodes_in[0]->data += nodes_out[i]->data;
      }
    }
  }
}; //class SplitLayer
} // namespace layer
} // namespace cxxnet
#endif
