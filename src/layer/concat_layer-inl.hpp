#ifndef CXXNET_LAYER_CONCAT_LAYER_INL_HPP_
#define CXXNET_LAYER_CONCAT_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"


namespace cxxnet {
namespace layer {

template<typename xpu, int dim>
class ConcatLayer : public ILayer<xpu> {
 public:
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() > 1 && nodes_out.size() == 1,
                 "Concat layer only support n-1 connection");
    utils::Check(nodes_in.size() <= 4, "More than 4 input node is unspported");
    mshadow::Shape<4> oshape = nodes_in[0]->data.shape_;
    mshadow::index_t out_ch = 0;
    for (mshadow::index_t i = 0; i < nodes_in.size(); ++i) {
      out_ch += nodes_in[i]->data.shape_[dim];
      for (mshadow::index_t j = 0; j < 4; ++j) {
        if (j == dim) continue;
        utils::Check(nodes_in[i]->data.shape_[j] == oshape[j],
                     "Concat shape doesn't match");
      }
    }
    oshape[dim] = out_ch;
    nodes_out[0]->data.shape_ = oshape;
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    switch(nodes_in.size()) {
    case 2:
      nodes_out[0]->data = concat<dim>(nodes_in[0]->data, nodes_in[1]->data);
      break;
    case 3:
      nodes_out[0]->data = concat<dim>(nodes_in[0]->data,
                                     concat<dim>(nodes_in[1]->data, nodes_in[2]->data));
      break;
    case 4:
      nodes_out[0]->data = concat<dim>(concat<dim>(nodes_in[0]->data, nodes_in[1]->data),
                                     concat<dim>(nodes_in[2]->data, nodes_in[3]->data));
      break;
    default:
      utils::Error("Too many node to concat");
      break;
    };
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (prop_grad) {
      switch(nodes_in.size()) {
      case 2:
        concat<dim>(nodes_in[0]->data, nodes_in[1]->data) = nodes_out[0]->data;
        break;
      case 3:
        concat<dim>(nodes_in[0]->data,
                  concat<dim>(nodes_in[1]->data, nodes_in[2]->data)) = nodes_out[0]->data;
        break;
      case 4:
        concat<dim>(concat<dim>(nodes_in[0]->data, nodes_in[1]->data),
                  concat<dim>(nodes_in[2]->data, nodes_in[3]->data)) = nodes_out[0]->data;
        break;
      default:
        utils::Error("Too many nodes to concat");
        break;
      };
    }
  }
}; //class ConcatLayer
} // namespace layer
} // namespace cxxnet
#endif
