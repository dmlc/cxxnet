#ifndef CXXNET_LAYER_MAXOUT_LAYER_INL_HPP_
#define CXXNET_LAYER_MAXOUT_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace cxxnet {
namespace layer {

template<typename Reducer, typename xpu>
class MaxoutLayer : public ILayer<xpu> {
 public:
  MaxoutLayer() {
    num_maxout_ = 1;
    mode_ = 0;
  }
  virtual ~MaxoutLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
    if (!strcmp(name, "maxout_unit")) num_maxout_ = atoi(val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    InitNode(nodes_in, nodes_out, p_cstate);
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    p_cstate->states[0].Resize(nodes_out[0]->data.shape_);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    if (mode_ == 0) {
      mshadow::Shape<2> pshape = nodes_out[0]->data[0][0].shape_;
      tmp = pool<Reducer>(nodes_in[0]->data, pshape, 1, num_maxout_, num_maxout_);
    } else {
      tmp = chpool<Reducer>(nodes_in[0]->data, 1, num_maxout_, 0);
    }
    mshadow::Copy(nodes_out[0]->data, tmp, nodes_out[0]->data.stream_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    if (prop_grad) {
      if (mode_ == 0) {
        nodes_in[0]->data = unpool<Reducer>(nodes_in[0]->data, tmp,
                                            nodes_out[0]->data, 1, num_maxout_, num_maxout_);
      } else {
        nodes_in[0]->data = ch_unpool<Reducer>(nodes_in[0]->data, tmp,
                                               nodes_out[0]->data, 1, num_maxout_, 0);
      }
    }
  }

 protected:
  inline void InitNode(const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "MaxoutLayer: only support 1-1 connection");
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    if (ishape[1] != 1) mode_ = 1;
    if (mode_ == 0) utils::Check(ishape[3] % num_maxout_ == 0,
                                 "hidden unit must divided by maxout unit");
    else utils::Check(ishape[1] % num_maxout_ == 0, "channle must divided by maxout unit");
    const int ch = mode_ == 0 ? 1 : ishape[1] / num_maxout_;
    const int dim = mode_ == 0 ? (ishape[3] - 1) / num_maxout_ + 1 : ishape[3];
    mshadow::Shape<4> oshape = mshadow::Shape4(ishape[0], ch, ishape[2], dim);
    nodes_out[0]->data.shape_ = oshape;
    // use 1 temp state to store pooled result
    p_cstate->states.push_back(mshadow::TensorContainer<xpu,4>(false));
    p_cstate->states[0].Resize(oshape);
  }
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  int num_maxout_;
  int mode_;

};   // class PoolingLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_POOLING_LAYER_INL_HPP_

